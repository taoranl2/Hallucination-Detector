# FILE: unified_hallucination_detector.py
"""
Unified Hallucination Detection System
Combines TSV (Truthfulness Separator Vector), MI-based uncertainty, and semi-supervised learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
import ot  # Python Optimal Transport library
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HallucinationSample:
    """Data structure for samples"""
    query: str
    response: str
    label: Optional[int] = None  # 0: factual, 1: hallucinated, None: unlabeled
    features: Optional[np.ndarray] = None
    embedding: Optional[torch.Tensor] = None


class TruthfulnessSeparatorVector(nn.Module):
    """
    TSV implementation following "Steer LLM Latents for Hallucination Detection"
    """
    def __init__(self, hidden_dim: int, steering_strength: float = 5.0):
        super().__init__()
        # Initialize as a learnable parameter
        self.v = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.steering_strength = steering_strength
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply TSV to hidden states"""
        # Ensure the steering vector is properly normalized if needed
        return hidden_states + self.steering_strength * self.v
    
    
class UnifiedHallucinationDetector:
    """
    Unified system combining multiple hallucination detection approaches:
    1. TSV for white-box latent space steering
    2. MI-based epistemic uncertainty quantification
    3. Optimal transport pseudo-labeling
    4. Meta-classifier combining all signals
    """
    
    def __init__(self, 
                 model_fn=None,
                 hidden_dim: int = 4096,
                 n_samples: int = 5,
                 temperature: float = 0.9,
                 use_tsv: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.model_fn = model_fn
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.temperature = temperature
        self.use_tsv = use_tsv
        self.device = device
        
        # Class prototypes for von Mises-Fisher distribution
        self.prototypes = {
            'truthful': None,
            'hallucinated': None
        }
        
        # Initialize TSV
        if use_tsv:
            self.tsv = TruthfulnessSeparatorVector(hidden_dim).to(device)
            # Initialize prototypes as random normalized vectors (detached)
            with torch.no_grad():
                self.prototypes['truthful'] = F.normalize(torch.randn(hidden_dim), dim=-1).to(device).detach()
                self.prototypes['hallucinated'] = F.normalize(torch.randn(hidden_dim), dim=-1).to(device).detach()
        else:
            self.tsv = None
        
        # Meta-classifier
        self.meta_classifier = None
        
    def compute_mi_uncertainty(self, query: str, responses: List[str], 
                            log_probs: List[float]) -> float:
        """Simplified MI-based uncertainty"""
        if len(responses) < 2:
            return 0.0
        
        # Just use entropy of response distribution as proxy
        probs = np.exp(log_probs)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def apply_tsv_and_extract_features(self, sample: HallucinationSample, 
                                     model=None, requires_grad: bool = False) -> torch.Tensor:
        """Apply TSV to model hidden states and extract features"""
        if not self.use_tsv:
            # Return mock embedding for black-box case
            embedding = torch.randn(self.hidden_dim).to(self.device)
            return embedding.detach() if not requires_grad else embedding
            
        # For TSV mode
        if model is None:
            # Mock hidden states for demonstration
            hidden_states = self._get_hidden_states(None, sample.query, sample.response)
            if requires_grad:
                hidden_states = hidden_states.requires_grad_(True)
        else:
            # Get real model hidden states
            with torch.no_grad() if not requires_grad else torch.enable_grad():
                hidden_states = self._get_hidden_states(model, sample.query, sample.response)
                
        # Apply TSV (this should maintain gradients if requires_grad=True)
        steered_states = self.tsv(hidden_states)
        
        # Extract last token embedding
        embedding = steered_states[-1]  # Last token
        
        return F.normalize(embedding, dim=-1)  # Normalize for vMF distribution
    
    def compute_vmf_probability(self, embedding: torch.Tensor, 
                               prototype: torch.Tensor, kappa: float = 10.0) -> float:
        """
        Compute von Mises-Fisher probability for class assignment
        """
        if prototype is None:
            return 0.5
        
        with torch.no_grad():
            similarity = torch.dot(embedding, prototype)
            return torch.exp(kappa * similarity).item()
    
    def optimal_transport_labeling(self, unlabeled_samples: List[HallucinationSample],
                                 class_distribution: np.ndarray) -> Dict[int, float]:
        """
        Assign pseudo-labels using optimal transport following TSV paper
        """
        n_samples = len(unlabeled_samples)
        
        # Compute probability matrix P
        P = np.zeros((n_samples, 2))
        
        for i, sample in enumerate(unlabeled_samples):
            if sample.embedding is not None and self.prototypes['truthful'] is not None:
                with torch.no_grad():
                    P[i, 0] = self.compute_vmf_probability(
                        sample.embedding, self.prototypes['truthful']
                    )
                    P[i, 1] = self.compute_vmf_probability(
                        sample.embedding, self.prototypes['hallucinated']
                    )
            else:
                # Fallback to uniform if embeddings not available
                P[i, :] = 0.5
                
        # Normalize rows
        P = P / (P.sum(axis=1, keepdims=True) + 1e-10)
        
        # Solve optimal transport problem
        M = -np.log(P + 1e-10)  # Cost matrix
        a = np.ones(n_samples) / n_samples  # Uniform source distribution
        b = class_distribution  # Target class distribution
        
        # Use entropic regularization
        Q = ot.sinkhorn(a, b, M, reg=0.05)
        
        # Extract pseudo-labels
        pseudo_labels = {}
        for i in range(n_samples):
            # Probability of being hallucinated
            pseudo_labels[i] = Q[i, 1] / Q[i, :].sum()
            
        return pseudo_labels
    
    def extract_comprehensive_features(self, sample: HallucinationSample) -> np.ndarray:
        """Extract all features for meta-classifier"""
        features = []
        
        # Sample multiple responses
        responses, log_probs = self._sample_responses(sample.query, self.n_samples)
        
        # 1. MI-based epistemic uncertainty
        mi_score = self.compute_mi_uncertainty(sample.query, responses, log_probs)
        features.append(mi_score)
        
        # 2. Semantic entropy
        semantic_entropy = self._compute_semantic_entropy(responses, log_probs)
        features.append(semantic_entropy)
        
        # 3. Self-consistency
        consistency = self._compute_self_consistency(responses)
        features.append(consistency)
        
        # 4. Response probability features
        primary_idx = responses.index(sample.response) if sample.response in responses else -1
        if primary_idx >= 0:
            features.append(log_probs[primary_idx])
        else:
            features.append(np.mean(log_probs))
            
        features.append(np.mean(log_probs))
        features.append(np.std(log_probs))
        
        # 5. Length features
        features.append(len(sample.response.split()))
        features.append(len(sample.query.split()))
        features.append(len(sample.response.split()) / (len(sample.query.split()) + 1))
        
        # 6. TSV-based features (if available)
        if self.use_tsv and sample.embedding is not None:
            # Distance to prototypes
            if self.prototypes['truthful'] is not None:
                with torch.no_grad():
                    truth_sim = torch.cosine_similarity(
                        sample.embedding.unsqueeze(0), 
                        self.prototypes['truthful'].unsqueeze(0)
                    ).item()
                    hall_sim = torch.cosine_similarity(
                        sample.embedding.unsqueeze(0),
                        self.prototypes['hallucinated'].unsqueeze(0)
                    ).item()
                features.extend([truth_sim, hall_sim, truth_sim - hall_sim])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
            
        return np.array(features)
    def save(self, filepath: str):
        """Save the trained detector to a file"""
        import pickle
        
        # Prepare state dict
        state = {
            'hidden_dim': self.hidden_dim,
            'n_samples': self.n_samples,
            'temperature': self.temperature,
            'use_tsv': self.use_tsv,
            'device': str(self.device),
            'prototypes': self.prototypes,
            'meta_classifier': self.meta_classifier,
            'tsv_state': self.tsv.state_dict() if self.tsv is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str, model_fn=None):
        """Load a trained detector from a file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance
        detector = cls(
            model_fn=model_fn,
            hidden_dim=state['hidden_dim'],
            n_samples=state['n_samples'],
            temperature=state['temperature'],
            use_tsv=state['use_tsv'],
            device=state['device']
        )
        
        # Restore state
        detector.prototypes = state['prototypes']
        detector.meta_classifier = state['meta_classifier']
        
        # Restore TSV if it was used
        if state['tsv_state'] is not None and detector.tsv is not None:
            detector.tsv.load_state_dict(state['tsv_state'])
        
        print(f"Model loaded from: {filepath}")
        return detector
    
    def train(self, 
              samples: List[HallucinationSample],
              n_initial_epochs: int = 5,
              n_augmented_epochs: int = 5,
              k_confident: int = 32):
        """
        Two-stage training following TSV paper
        """
        # Separate labeled and unlabeled
        labeled_samples = [s for s in samples if s.label is not None]
        unlabeled_samples = [s for s in samples if s.label is None]
        
        print(f"Training with {len(labeled_samples)} labeled and {len(unlabeled_samples)} unlabeled samples")
        
        # Stage 1: Initial training on labeled data
        if self.use_tsv and len(labeled_samples) > 0:
            try:
                self._train_tsv_initial(labeled_samples, n_initial_epochs)
            except RuntimeError as e:
                print(f"Warning: TSV training failed with error: {e}")
                print("Continuing without TSV training...")
                self.use_tsv = False
            
        # Extract features for all samples
        print("Extracting features...")
        for i, sample in enumerate(tqdm(samples)):
            if self.use_tsv:
                # Extract embedding without gradients for feature extraction
                sample.embedding = self.apply_tsv_and_extract_features(sample, requires_grad=False)
            sample.features = self.extract_comprehensive_features(sample)
            
        # Stage 2: Pseudo-labeling and augmented training
        if len(unlabeled_samples) > 0:
            # Estimate class distribution from labeled data
            class_dist = np.array([
                sum(1 for s in labeled_samples if s.label == 0),
                sum(1 for s in labeled_samples if s.label == 1)
            ])
            class_dist = class_dist / class_dist.sum()
            
            # Optimal transport pseudo-labeling
            pseudo_labels = self.optimal_transport_labeling(unlabeled_samples, class_dist)
            
            # Select confident samples
            confidence_scores = []
            for i, sample in enumerate(unlabeled_samples):
                # Compute confidence based on entropy
                p_hall = pseudo_labels[i]
                p_truth = 1 - p_hall
                entropy = -p_hall * np.log(p_hall + 1e-10) - p_truth * np.log(p_truth + 1e-10)
                confidence_scores.append(1 - entropy)
                
            # Select top-k confident samples
            confident_indices = np.argsort(confidence_scores)[-k_confident:]
            
            # Augment training set
            augmented_samples = labeled_samples.copy()
            for idx in confident_indices:
                sample = unlabeled_samples[idx]
                sample.label = 1 if pseudo_labels[idx] > 0.5 else 0
                augmented_samples.append(sample)
                
            print(f"Augmented with {len(confident_indices)} confident pseudo-labeled samples")
            
            # Retrain TSV if applicable and still enabled
            if self.use_tsv and hasattr(self, 'tsv') and self.tsv is not None:
                try:
                    self._train_tsv_augmented(augmented_samples, n_augmented_epochs)
                except RuntimeError as e:
                    print(f"Warning: Augmented TSV training failed with error: {e}")
                    print("Continuing without augmented TSV training...")
                
        # Train meta-classifier
        self._train_meta_classifier(labeled_samples, unlabeled_samples)
        
    def _train_tsv_initial(self, labeled_samples: List[HallucinationSample], 
                          n_epochs: int):
        """Initial TSV training"""
        optimizer = torch.optim.AdamW(self.tsv.parameters(), lr=5e-3)
        
        for epoch in range(n_epochs):
            total_loss = 0
            
            for sample in labeled_samples:
                optimizer.zero_grad()
                
                # Get embedding with TSV (with gradients enabled)
                embedding = self.apply_tsv_and_extract_features(sample, requires_grad=True)
                
                # Update prototypes (exponential moving average)
                label_key = 'truthful' if sample.label == 0 else 'hallucinated'
                with torch.no_grad():
                    if self.prototypes[label_key] is None:
                        self.prototypes[label_key] = embedding.clone().detach()
                        self.prototypes[label_key] = F.normalize(self.prototypes[label_key], dim=-1)
                    else:
                        self.prototypes[label_key] = 0.99 * self.prototypes[label_key] + 0.01 * embedding.detach()
                        self.prototypes[label_key] = F.normalize(self.prototypes[label_key], dim=-1)
                
                # Compute vMF loss
                loss = self._compute_vmf_loss(embedding, sample.label)
                
                if loss.requires_grad:  # Only backprop if gradients are available
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(labeled_samples):.4f}")
                
    def _compute_vmf_loss(self, embedding: torch.Tensor, label: int, kappa: float = 10.0) -> torch.Tensor:
        """von Mises-Fisher loss"""
        # Initialize prototypes if they don't exist
        if self.prototypes['truthful'] is None:
            with torch.no_grad():
                self.prototypes['truthful'] = torch.randn(self.hidden_dim).to(self.device)
                self.prototypes['truthful'] = F.normalize(self.prototypes['truthful'], dim=-1)
        if self.prototypes['hallucinated'] is None:
            with torch.no_grad():
                self.prototypes['hallucinated'] = torch.randn(self.hidden_dim).to(self.device)
                self.prototypes['hallucinated'] = F.normalize(self.prototypes['hallucinated'], dim=-1)
            
        # Compute probabilities (prototypes should be detached)
        p_truth = torch.exp(kappa * torch.dot(embedding, self.prototypes['truthful'].detach()))
        p_hall = torch.exp(kappa * torch.dot(embedding, self.prototypes['hallucinated'].detach()))
        
        # Normalize
        p_total = p_truth + p_hall
        p_truth = p_truth / p_total
        p_hall = p_hall / p_total
        
        # Cross-entropy loss
        if label == 0:  # Truthful
            loss = -torch.log(p_truth + 1e-10)
        else:  # Hallucinated
            loss = -torch.log(p_hall + 1e-10)
            
        return loss
    
    def _train_meta_classifier(self, labeled_samples: List[HallucinationSample],
                              unlabeled_samples: List[HallucinationSample]):
        """Train the meta-classifier using all features"""
        # Prepare data
        X_labeled = np.array([s.features for s in labeled_samples])
        y_labeled = np.array([s.label for s in labeled_samples])
        
        # Base classifier
        base_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        if len(unlabeled_samples) > 0:
            # Semi-supervised learning
            X_unlabeled = np.array([s.features for s in unlabeled_samples])
            
            self.meta_classifier = SelfTrainingClassifier(
                base_clf,
                threshold=0.75,
                criterion='threshold',
                max_iter=10
            )
            
            X_all = np.vstack([X_labeled, X_unlabeled])
            y_all = np.concatenate([y_labeled, [-1] * len(unlabeled_samples)])
            
            self.meta_classifier.fit(X_all, y_all)
        else:
            # Supervised only
            self.meta_classifier = base_clf
            self.meta_classifier.fit(X_labeled, y_labeled)
            
        # Feature importance
        feature_names = [
            'MI Score', 'Semantic Entropy', 'Self Consistency',
            'Primary Log Prob', 'Mean Log Prob', 'Std Log Prob',
            'Response Length', 'Query Length', 'Length Ratio',
            'Truth Similarity', 'Hall Similarity', 'Similarity Diff'
        ]
        
        importances = self.meta_classifier.feature_importances_ if hasattr(
            self.meta_classifier, 'feature_importances_'
        ) else self.meta_classifier.base_estimator_.feature_importances_
        
        print("\nFeature Importances:")
        for name, imp in sorted(zip(feature_names, importances), 
                               key=lambda x: x[1], reverse=True):
            print(f"{name}: {imp:.3f}")
            
    def predict(self, query: str, response: str) -> Tuple[int, float, Dict[str, float]]:
        """
        Predict hallucination with detailed scores
        Returns: (prediction, confidence, component_scores)
        """
        sample = HallucinationSample(query=query, response=response)
        
        # Extract all features
        if self.use_tsv:
            sample.embedding = self.apply_tsv_and_extract_features(sample, requires_grad=False)
        sample.features = self.extract_comprehensive_features(sample)
        
        # Get prediction from meta-classifier
        if self.meta_classifier is None:
            raise ValueError("Model not trained!")
            
        prediction = self.meta_classifier.predict(sample.features.reshape(1, -1))[0]
        confidence = self.meta_classifier.predict_proba(sample.features.reshape(1, -1))[0, 1]
        
        # Component scores for interpretability
        component_scores = {
            'mi_score': sample.features[0],
            'semantic_entropy': sample.features[1],
            'self_consistency': sample.features[2],
            'tsv_score': sample.features[11] if self.use_tsv else 0.0
        }
        
        return prediction, confidence, component_scores
    
    # Helper methods
    def _sample_responses(self, query: str, n: int) -> Tuple[List[str], List[float]]:
        """Sample n responses from the model"""
        if self.model_fn is None:
            # Mock for testing
            mock_responses = [f"Response {i} to: {query[:30]}..." for i in range(n)]
            mock_log_probs = np.random.uniform(-5, -0.1, n).tolist()
            return mock_responses, mock_log_probs
            
        responses = []
        log_probs = []
        
        for _ in range(n):
            resp, log_prob = self.model_fn(query, temperature=self.temperature)
            responses.append(resp)
            log_probs.append(log_prob)
            
        return responses, log_probs
    
    def _cluster_responses(self, responses: List[str], log_probs: List[float],
                          threshold: float = 0.25) -> Dict[str, Tuple[List[int], float]]:
        """Cluster similar responses based on F1 score"""
        clusters = {}
        used = set()
        
        for i, (resp1, prob1) in enumerate(zip(responses, log_probs)):
            if i in used:
                continue
                
            cluster_indices = [i]
            cluster_prob = np.exp(prob1)
            
            for j, (resp2, prob2) in enumerate(zip(responses[i+1:], log_probs[i+1:]), i+1):
                if j in used:
                    continue
                    
                if self._f1_score(resp1, resp2) >= threshold:
                    cluster_indices.append(j)
                    cluster_prob += np.exp(prob2)
                    used.add(j)
                    
            clusters[resp1] = (cluster_indices, cluster_prob)
            used.add(i)
            
        return clusters
    
    def _f1_score(self, text1: str, text2: str) -> float:
        """Token-based F1 score"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1 & tokens2)
        precision = intersection / len(tokens2)
        recall = intersection / len(tokens1)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
    
    def _compute_semantic_entropy(self, responses: List[str], log_probs: List[float]) -> float:
        """Compute entropy over semantically clustered responses"""
        clusters = self._cluster_responses(responses, log_probs)
        
        Z = sum(p for _, p in clusters.values())
        entropy = 0.0
        
        for _, (_, p) in clusters.items():
            p_norm = p / Z
            if p_norm > 0:
                entropy -= p_norm * np.log(p_norm)
                
        return entropy
    
    def _compute_self_consistency(self, responses: List[str]) -> float:
        """Compute consistency score"""
        if len(responses) <= 1:
            return 1.0
            
        clusters = self._cluster_responses(responses, [1.0] * len(responses))
        return 1.0 / len(clusters)
    
    def _get_hidden_states(self, model, query: str, response: str) -> torch.Tensor:
        """Get hidden states from model - this is a placeholder"""
        # In practice, this would hook into the actual model layers
        # For now, return mock hidden states
        seq_len = len((query + " " + response).split())
        hidden_states = torch.randn(seq_len, self.hidden_dim).to(self.device)
        return hidden_states
    
    def _train_tsv_augmented(self, augmented_samples: List[HallucinationSample], 
                            n_epochs: int):
        """Continue training TSV with augmented data"""
        # Similar to initial training but with augmented dataset
        self._train_tsv_initial(augmented_samples, n_epochs)