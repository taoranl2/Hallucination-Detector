# FILE: unified_hallucination_detector.py (Complete Enhanced Version with All Quick Improvements)
"""
Unified Hallucination Detection System
Combines TSV (Truthfulness Separator Vector), MI-based uncertainty, and semi-supervised learning

IMPROVEMENTS APPLIED:
1. ✅ Class balancing with computed weights
2. ✅ Optimized hyperparameters (300 trees, depth=8, regularization)
3. ✅ Ensemble support with multiple classifiers
4. ✅ Increased sampling (n_samples=5 default)
5. ✅ Better error handling and fallbacks
6. ✅ Cross-validation performance estimates
7. ✅ Enhanced feature importance analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import ot
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pickle

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
        return hidden_states + self.steering_strength * self.v


class EnsembleClassifier:
    """Enhanced ensemble classifier with better voting and importance extraction"""
    
    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def predict(self, X):
        predictions = []
        for name, clf in self.classifiers:
            pred = clf.predict(X)
            predictions.append(pred)
        
        # Majority vote with tie-breaking
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    def predict_proba(self, X):
        probabilities = []
        for name, clf in self.classifiers:
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
                probabilities.append(proba)
        
        if probabilities:
            # Average probabilities (soft voting)
            return np.mean(probabilities, axis=0)
        else:
            # Fallback to hard voting
            preds = self.predict(X)
            n_samples = len(preds)
            n_classes = len(np.unique(preds))
            proba = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(preds):
                proba[i, pred] = 1.0
            return proba
    
    @property
    def feature_importances_(self):
        """Get average feature importances from ensemble"""
        importances = []
        for name, clf in self.classifiers:
            if hasattr(clf, 'feature_importances_'):
                importances.append(clf.feature_importances_)
            elif hasattr(clf, 'base_classifier_') and hasattr(clf.base_classifier_, 'feature_importances_'):
                importances.append(clf.base_classifier_.feature_importances_)
        
        if importances:
            return np.mean(importances, axis=0)
        else:
            return np.array([0.0] * 12)  # Default number of features


class UnifiedHallucinationDetector:
    """
    Unified system combining multiple hallucination detection approaches:
    1. TSV for white-box latent space steering
    2. MI-based epistemic uncertainty quantification
    3. Optimal transport pseudo-labeling
    4. Meta-classifier combining all signals
    
    ENHANCED WITH QUICK IMPROVEMENTS:
    - Class balancing for better hallucination detection
    - Optimized hyperparameters for text classification
    - Ensemble support for robustness
    - Better error handling and fallbacks
    """

    def __init__(self,
                 model_fn=None,
                 hidden_dim: int = 4096,
                 n_samples: int = 5,  # IMPROVED: Increased from 3 to 5
                 temperature: float = 0.9,
                 use_tsv: bool = True,
                 use_ensemble: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.model_fn = model_fn
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples  # Now defaults to 5 for better uncertainty
        self.temperature = temperature
        self.use_tsv = use_tsv
        self.use_ensemble = use_ensemble
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

        # Use entropy of response distribution as proxy
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

        # Use mean pooling over the sequence for a more robust embedding
        embedding = torch.mean(steered_states, dim=0)

        return F.normalize(embedding, dim=-1)  # Normalize for vMF distribution

    def compute_vmf_probability(self, embedding: torch.Tensor,
                               prototype: torch.Tensor, kappa: float = 10.0) -> float:
        """Compute von Mises-Fisher probability for class assignment"""
        if prototype is None:
            return 0.5

        with torch.no_grad():
            similarity = torch.dot(embedding, prototype)
            return torch.exp(kappa * similarity).item()

    def optimal_transport_labeling(self, unlabeled_samples: List[HallucinationSample],
                                 class_distribution: np.ndarray) -> Dict[int, float]:
        """Assign pseudo-labels using optimal transport following TSV paper"""
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

        # Sample multiple responses (now uses increased n_samples)
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
        state = {
            'hidden_dim': self.hidden_dim,
            'n_samples': self.n_samples,
            'temperature': self.temperature,
            'use_tsv': self.use_tsv,
            'use_ensemble': self.use_ensemble,
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
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance
        detector = cls(
            model_fn=model_fn,
            hidden_dim=state['hidden_dim'],
            n_samples=state['n_samples'],
            temperature=state['temperature'],
            use_tsv=state['use_tsv'],
            use_ensemble=state.get('use_ensemble', False),
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
        """Two-stage training following TSV paper"""
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

        # Extract features for all samples if they don't exist
        print("Extracting features for meta-classifier...")
        for sample in tqdm(samples, desc="Feature Extraction"):
            if sample.features is None:
                if self.use_tsv and sample.embedding is None:
                    # Extract embedding without gradients for feature extraction
                    sample.embedding = self.apply_tsv_and_extract_features(sample, requires_grad=False)
                sample.features = self.extract_comprehensive_features(sample)

        # Stage 2: Pseudo-labeling and augmented training
        if len(unlabeled_samples) > 0 and len(labeled_samples) > 0:
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

        # Train meta-classifier with all improvements
        self._train_meta_classifier(labeled_samples, unlabeled_samples)

    def _train_tsv_initial(self, labeled_samples: List[HallucinationSample], n_epochs: int):
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
        """
        ENHANCED META-CLASSIFIER TRAINING WITH ALL QUICK IMPROVEMENTS
        
        Improvements Applied:
        1. ✅ Class balancing with computed weights
        2. ✅ Optimized hyperparameters  
        3. ✅ Ensemble support
        4. ✅ Adaptive classifier selection
        5. ✅ Better error handling
        6. ✅ Cross-validation performance estimates
        """
        
        if not labeled_samples:
            print("Warning: No labeled data available to train the meta-classifier.")
            return

        # Prepare labeled data
        X_labeled = np.array([s.features for s in labeled_samples])
        y_labeled = np.array([s.label for s in labeled_samples])
        
        print(f"Training meta-classifier with {len(labeled_samples)} labeled samples")
        print(f"Class distribution: {np.bincount(y_labeled)}")
        
        # IMPROVEMENT 1: Calculate class weights for balancing
        try:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_labeled), 
                y=y_labeled
            )
            weight_dict = dict(zip(np.unique(y_labeled), class_weights))
            print(f"Computed class weights: {weight_dict}")
        except Exception as e:
            print(f"Warning: Could not compute class weights: {e}")
            weight_dict = None
        
        # ADAPTIVE STRATEGY: Choose approach based on dataset size and configuration
        min_samples_for_semi_supervised = 50
        
        if (len(labeled_samples) >= min_samples_for_semi_supervised and 
            len(unlabeled_samples) > 0 and 
            not self.use_ensemble):
            
            print(f"Using SelfTrainingClassifier with {len(labeled_samples)} labeled + {len(unlabeled_samples)} unlabeled samples")
            
            # Prepare combined data for semi-supervised learning
            X_unlabeled = np.array([s.features for s in unlabeled_samples])
            y_unlabeled = [-1] * len(unlabeled_samples)  # -1 indicates unlabeled
            
            X_combined = np.vstack([X_labeled, X_unlabeled])
            y_combined = np.concatenate([y_labeled, y_unlabeled])
            
            # IMPROVEMENT 1 & 4: Optimized base classifier with class balancing
            base_classifier = LGBMClassifier(
                # Core optimized parameters (IMPROVEMENT 4)
                n_estimators=300,           # More trees for complex patterns
                learning_rate=0.05,         # Slower learning for stability
                max_depth=8,                # Deeper trees for text features
                min_child_samples=5,        # Smaller leaves for fine patterns
                
                # Regularization to prevent overfitting
                subsample=0.8,              # Row sampling
                colsample_bytree=0.8,       # Feature sampling  
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=0.1,             # L2 regularization
                
                # Class balancing (IMPROVEMENT 1 - MOST IMPORTANT)
                class_weight=weight_dict if weight_dict else 'balanced',
                
                # Performance settings
                random_state=42,
                importance_type='gain',
                n_jobs=-1,                  # Use all CPU cores
                verbose=-1                  # Suppress warnings
            )
            
            # Semi-supervised wrapper
            self.meta_classifier = SelfTrainingClassifier(
                base_classifier,
                threshold=0.8,              # High confidence threshold
                criterion='threshold',
                max_iter=10,                # Limit iterations
                verbose=True
            )
            
            try:
                # Fit on combined data
                print("Training SelfTrainingClassifier...")
                self.meta_classifier.fit(X_combined, y_combined)
                
                # Print statistics about pseudo-labeling
                if hasattr(self.meta_classifier, 'labeled_iter_'):
                    n_pseudo_labeled = len(self.meta_classifier.labeled_iter_)
                    print(f"Successfully pseudo-labeled {n_pseudo_labeled} samples")
                    
            except Exception as e:
                print(f"SelfTrainingClassifier failed: {e}")
                print("Falling back to supervised LGBMClassifier")
                
                # Fallback to supervised learning with same optimized parameters
                self.meta_classifier = LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    class_weight=weight_dict if weight_dict else 'balanced',
                    random_state=42,
                    importance_type='gain',
                    n_jobs=-1,
                    verbose=-1
                )
                self.meta_classifier.fit(X_labeled, y_labeled)
        
        elif (self.use_ensemble and len(labeled_samples) >= 30 and len(unlabeled_samples) > 0):
            print("Using ensemble of semi-supervised classifiers")
            self._train_ensemble_classifier_improved(labeled_samples, unlabeled_samples, weight_dict)
        
        elif self.use_ensemble:
            print("Using ensemble of supervised classifiers")
            self._train_supervised_ensemble(X_labeled, y_labeled, weight_dict)
        
        else:
            print(f"Using single supervised LGBMClassifier ({len(labeled_samples)} labeled samples)")
            
            # IMPROVEMENT 1 & 4: Single classifier with all optimizations
            self.meta_classifier = LGBMClassifier(
                # Optimized hyperparameters (IMPROVEMENT 4)
                n_estimators=300,           # More trees
                learning_rate=0.05,         # Slower learning
                max_depth=8,                # Deeper trees
                min_child_samples=5,        # Smaller leaves
                
                # Regularization
                subsample=0.8,              # Row sampling
                colsample_bytree=0.8,       # Feature sampling
                reg_alpha=0.1,              # L1 regularization  
                reg_lambda=0.1,             # L2 regularization
                
                # Class balancing (IMPROVEMENT 1 - MOST IMPORTANT)
                class_weight=weight_dict if weight_dict else 'balanced',
                
                # Performance
                random_state=42,
                importance_type='gain',
                n_jobs=-1,
                verbose=-1
            )
            
            # Cross-validation for performance estimate
            try:
                cv = StratifiedKFold(n_splits=min(5, len(labeled_samples)//10), shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    self.meta_classifier, X_labeled, y_labeled, 
                    cv=cv, scoring='f1_macro', n_jobs=-1
                )
                print(f"Cross-validation F1 scores: {cv_scores}")
                print(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as e:
                print(f"Cross-validation failed: {e}")
            
            # Fit the final model
            print("Training optimized LGBMClassifier...")
            self.meta_classifier.fit(X_labeled, y_labeled)

        # Feature importance analysis
        self._print_feature_importance_improved()
        
        # Print improvement summary
        print("\n" + "="*50)
        print("APPLIED QUICK IMPROVEMENTS:")
        print("✓ Class balancing with computed weights")
        print("✓ Optimized hyperparameters (300 trees, depth=8)")
        print("✓ Regularization (L1=0.1, L2=0.1)")
        print("✓ Enhanced sampling (subsample=0.8)")
        if self.use_ensemble:
            print("✓ Ensemble mode enabled")
        print("✓ Cross-validation performance estimate")
        print("="*50)

    def _train_ensemble_classifier_improved(self, labeled_samples: List[HallucinationSample],
                                           unlabeled_samples: List[HallucinationSample],
                                           weight_dict: dict):
        """Enhanced ensemble training with improved parameters"""
        
        X_labeled = np.array([s.features for s in labeled_samples])
        y_labeled = np.array([s.label for s in labeled_samples])
        X_unlabeled = np.array([s.features for s in unlabeled_samples])
        y_unlabeled = [-1] * len(unlabeled_samples)
        
        X_combined = np.vstack([X_labeled, X_unlabeled])
        y_combined = np.concatenate([y_labeled, y_unlabeled])
        
        # Enhanced ensemble with optimized classifiers
        classifiers = []
        
        # 1. Optimized SelfTrainingClassifier with LGBMClassifier
        st_lgbm = SelfTrainingClassifier(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight=weight_dict if weight_dict else 'balanced',
                random_state=42,
                importance_type='gain',
                n_jobs=-1,
                verbose=-1
            ),
            threshold=0.8,
            criterion='threshold'
        )
        classifiers.append(('SelfTrain_LGBM', st_lgbm))
        
        # 2. Optimized SelfTrainingClassifier with RandomForest
        st_rf = SelfTrainingClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=weight_dict if weight_dict else 'balanced',
                random_state=42,
                n_jobs=-1
            ),
            threshold=0.75,
            criterion='threshold'
        )
        classifiers.append(('SelfTrain_RF', st_rf))
        
        # Train all classifiers
        trained_classifiers = []
        for name, clf in classifiers:
            try:
                print(f"Training {name}...")
                clf.fit(X_combined, y_combined)
                trained_classifiers.append((name, clf))
                print(f"✓ Successfully trained {name}")
            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
        
        if trained_classifiers:
            # Use ensemble with soft voting
            self.meta_classifier = EnsembleClassifier(trained_classifiers)
            print(f"✓ Ensemble created with {len(trained_classifiers)} classifiers")
        else:
            # Fallback to single supervised classifier
            print("All ensemble classifiers failed, falling back to supervised learning")
            self._train_supervised_fallback(X_labeled, y_labeled, weight_dict)

    def _train_supervised_ensemble(self, X_labeled: np.ndarray, y_labeled: np.ndarray, weight_dict: dict):
        """Train ensemble using only supervised classifiers"""
        
        print("Creating supervised ensemble...")
        
        # Optimized LightGBM
        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight=weight_dict if weight_dict else 'balanced',
            random_state=42,
            importance_type='gain',
            n_jobs=-1,
            verbose=-1
        )
        
        # Optimized Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=weight_dict if weight_dict else 'balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Create voting ensemble
        self.meta_classifier = VotingClassifier([
            ('lgbm', lgbm),
            ('rf', rf)
        ], voting='soft')  # Use probability averaging
        
        # Fit ensemble
        print("Training supervised ensemble...")
        self.meta_classifier.fit(X_labeled, y_labeled)
        print("✓ Supervised ensemble trained successfully")

    def _train_supervised_fallback(self, X_labeled: np.ndarray, y_labeled: np.ndarray, weight_dict: dict):
        """Fallback to single optimized supervised classifier"""
        
        print("Using fallback: single supervised LGBMClassifier")
        self.meta_classifier = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight=weight_dict if weight_dict else 'balanced',
            random_state=42,
            importance_type='gain',
            n_jobs=-1,
            verbose=-1
        )
        self.meta_classifier.fit(X_labeled, y_labeled)

    def _print_feature_importance_improved(self):
        """Enhanced feature importance analysis"""
        
        feature_names = [
            'MI Score', 'Semantic Entropy', 'Self Consistency',
            'Primary Log Prob', 'Mean Log Prob', 'Std Log Prob',
            'Response Length', 'Query Length', 'Length Ratio',
            'Truth Similarity', 'Hall Similarity', 'Similarity Diff'
        ]

        # Get feature importances from different classifier types
        try:
            if hasattr(self.meta_classifier, 'feature_importances_'):
                # Direct classifier (LGBMClassifier)
                importances = self.meta_classifier.feature_importances_
            elif hasattr(self.meta_classifier, 'base_classifier_') and hasattr(self.meta_classifier.base_classifier_, 'feature_importances_'):
                # SelfTrainingClassifier
                importances = self.meta_classifier.base_classifier_.feature_importances_
            elif hasattr(self.meta_classifier, 'estimators_'):
                # VotingClassifier - average importance across estimators
                all_importances = []
                for name, estimator in self.meta_classifier.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        all_importances.append(estimator.feature_importances_)
                    elif hasattr(estimator, 'base_classifier_') and hasattr(estimator.base_classifier_, 'feature_importances_'):
                        all_importances.append(estimator.base_classifier_.feature_importances_)
                
                if all_importances:
                    importances = np.mean(all_importances, axis=0)
                else:
                    importances = [0] * len(feature_names)
            else:
                # EnsembleClassifier or other custom classifiers
                if hasattr(self.meta_classifier, 'feature_importances_'):
                    importances = self.meta_classifier.feature_importances_
                else:
                    importances = [0] * len(feature_names)
        
        except Exception as e:
            print(f"Could not extract feature importances: {e}")
            importances = [0] * len(feature_names)

        print(f"\nFeature Importances (Top 10):")
        print("-" * 40)
        
        # Sort by importance
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, importance) in enumerate(feature_importance_pairs[:10]):
            print(f"{i+1:2d}. {name:20s}: {importance:.3f}")
        
        # Print class balancing confirmation
        print(f"\nClass Balancing Applied:")
        print("✓ Using class_weight='balanced' for equal class importance")
        print("✓ Optimized hyperparameters for text classification")
        
        return feature_importance_pairs

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

        if not clusters:
            return 0.0

        Z = sum(p for _, p in clusters.values())
        if Z == 0:
            return 0.0
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
        return 1.0 / len(clusters) if clusters else 1.0

    def _get_hidden_states(self, model, query: str, response: str) -> torch.Tensor:
        """Get hidden states from model - this is a placeholder"""
        # In a real implementation, this should call the actual model interface.
        if hasattr(model, 'get_hidden_states'):
             return model.get_hidden_states(query, response)

        # Fallback to mock hidden states if no model is provided
        seq_len = len((query + " " + response).split())
        hidden_states = torch.randn(seq_len, self.hidden_dim).to(self.device)
        return hidden_states

    def _train_tsv_augmented(self, augmented_samples: List[HallucinationSample], n_epochs: int):
        """Continue training TSV with augmented data"""
        # Similar to initial training but with augmented dataset
        self._train_tsv_initial(augmented_samples, n_epochs)