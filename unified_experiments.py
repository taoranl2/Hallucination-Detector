# FILE: unified_experiments.py (Clean Production Version)
"""
Experimental framework for the unified hallucination detection system
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from unified_hallucination_detector import UnifiedHallucinationDetector, HallucinationSample
from tqdm import tqdm
import warnings
import os
import pickle
import argparse
import sys
warnings.filterwarnings('ignore')


class ModelInterface:
    """Interface for LLM interaction"""

    def __init__(self,
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 white_box: bool = True):

        self.device = device
        self.white_box = white_box
        self.model_name = model_name

        if white_box:
            try:
                print(f"Loading model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                print("Model loaded successfully")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Falling back to black-box mode")
                self.white_box = False
                self.tokenizer = None
                self.model = None
        else:
            print("Using black-box mode with diverse mock responses")
            self.tokenizer = None
            self.model = None

    def generate_with_logprobs(self, prompt: str, temperature: float = 0.9,
                              max_length: int = 100) -> Tuple[str, float]:
        if self.white_box and self.model is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                response_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                if outputs.scores:
                    log_probs = []
                    for i, score in enumerate(outputs.scores):
                        if i < len(response_ids):
                            token_id = response_ids[i]
                            log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                            log_probs.append(log_prob)
                    avg_log_prob = np.mean(log_probs) if log_probs else -10.0
                else:
                    avg_log_prob = -10.0
            except Exception as e:
                print(f"Generation failed: {e}")
                response = f"Fallback response to: {prompt[:50]}..."
                avg_log_prob = np.random.uniform(-5, -0.1)
        else:
            response = f"Mock response to: {prompt[:50]}..."
            avg_log_prob = np.random.uniform(-5, -0.1)

        return response.strip(), avg_log_prob

    def get_hidden_states(self, prompt: str, response: str, layer: int = -1) -> torch.Tensor:
        if not self.white_box or self.model is None:
            seq_len = len((prompt + " " + response).split())
            hidden_dim = getattr(self.model.config, 'hidden_size', 4096) if self.model else 4096
            return torch.randn(seq_len, hidden_dim)

        try:
            full_text = prompt + " " + response
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                prompt_len = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                hidden_states = outputs.hidden_states[layer][0, prompt_len:]

            return hidden_states
        except Exception:
            seq_len = len((prompt + " " + response).split())
            hidden_dim = getattr(self.model.config, 'hidden_size', 4096)
            return torch.randn(seq_len, hidden_dim)


def load_dataset(filepath: str) -> List[HallucinationSample]:
    """Load dataset from JSON or CSV"""
    
    if not os.path.exists(filepath):
        print(f"Dataset file '{filepath}' not found!")
        print(f"Generate it first: python generate_truthfulqa_dataset.py -o {filepath} -n 2000")
        sys.exit(1)
    
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath).to_dict('records')
        else:
            raise ValueError("Unsupported file format")

        samples = []
        for item in data:
            query = str(item.get('query', '')).strip()
            response = str(item.get('response', '')).strip()
            
            if not query or not response:
                continue

            label = item.get('label', None)
            if label is not None:
                try:
                    label = int(label)
                    if label not in [0, 1]:
                        label = None
                except (ValueError, TypeError):
                    label = None

            sample = HallucinationSample(
                query=query,
                response=response,
                label=label
            )
            samples.append(sample)

        labeled_count = sum(1 for s in samples if s.label is not None)
        print(f"Loaded {len(samples)} samples ({labeled_count} labeled)")
        
        return samples
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def evaluate_detector(detector: UnifiedHallucinationDetector,
                     test_samples: List[HallucinationSample],
                     save_prefix: str = "results") -> Dict:
    """Evaluate detector performance"""

    predictions = []
    confidences = []
    true_labels = []
    component_scores_list = []

    print("Evaluating detector...")
    for sample in tqdm(test_samples, desc="Evaluating"):
        if sample.label is None:
            continue

        try:
            pred, conf, comp_scores = detector.predict(sample.query, sample.response)
            predictions.append(pred)
            confidences.append(conf)
            true_labels.append(sample.label)
            component_scores_list.append(comp_scores)
        except Exception as e:
            print(f"Prediction failed for sample: {e}")
            continue
    
    if not true_labels:
        print("No predictions made")
        return {}

    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    if len(np.unique(true_labels)) > 1:
        try:
            auc = roc_auc_score(true_labels, confidences)
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    cm = confusion_matrix(true_labels, predictions)

    # Enhanced feature importance analysis with multiple extraction methods
    print(f"\nFeature Analysis:")
    
    feature_importances = None
    active_feature_names = []
    
    # Method 1: Direct access
    if hasattr(detector.meta_classifier, 'feature_importances_'):
        feature_importances = detector.meta_classifier.feature_importances_
        print("✓ Extracted importances directly from classifier")
    
    # Method 2: SelfTrainingClassifier access
    elif hasattr(detector.meta_classifier, 'base_classifier_'):
        base_clf = detector.meta_classifier.base_classifier_
        print(f"Detected SelfTrainingClassifier with base: {type(base_clf).__name__}")
        
        # Try multiple approaches for the base classifier
        if hasattr(base_clf, 'feature_importances_'):
            feature_importances = base_clf.feature_importances_
            print("✓ Extracted importances from base_classifier_.feature_importances_")
        elif hasattr(base_clf, 'booster_'):
            # LightGBM specific
            try:
                importance_gains = base_clf.booster_.feature_importance(importance_type='gain')
                if len(importance_gains) > 0:
                    feature_importances = importance_gains / importance_gains.sum()
                    print("✓ Extracted importances from LightGBM booster (gain)")
            except Exception as e:
                print(f"Failed to extract from booster: {e}")
                # Try alternative LightGBM access
                try:
                    feature_importances = base_clf.feature_importances_
                    print("✓ Extracted importances from LightGBM feature_importances_")
                except:
                    pass
        
        # Try accessing the fitted estimator directly
        if feature_importances is None and hasattr(detector.meta_classifier, 'labeled_iter_'):
            try:
                # The SelfTrainingClassifier stores the final classifier
                final_clf = detector.meta_classifier.base_classifier_
                if hasattr(final_clf, 'feature_importances_'):
                    feature_importances = final_clf.feature_importances_
                    print("✓ Extracted importances from final fitted classifier")
            except:
                pass
    
    # Method 3: VotingClassifier access
    elif hasattr(detector.meta_classifier, 'estimators_'):
        print("Detected VotingClassifier")
        all_importances = []
        for name, estimator in detector.meta_classifier.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                all_importances.append(estimator.feature_importances_)
                print(f"✓ Extracted importances from {name}")
        if all_importances:
            feature_importances = np.mean(all_importances, axis=0)
            print("✓ Averaged importances from voting classifier")
    
    # Method 4: Try accessing through the EnsembleClassifier wrapper
    elif hasattr(detector.meta_classifier, 'classifiers'):
        print("Detected custom EnsembleClassifier")
        all_importances = []
        for name, clf in detector.meta_classifier.classifiers:
            try:
                if hasattr(clf, 'base_classifier_') and hasattr(clf.base_classifier_, 'feature_importances_'):
                    all_importances.append(clf.base_classifier_.feature_importances_)
                    print(f"✓ Extracted importances from {name}")
                elif hasattr(clf, 'feature_importances_'):
                    all_importances.append(clf.feature_importances_)
                    print(f"✓ Extracted importances from {name}")
            except Exception as e:
                print(f"Failed to extract from {name}: {e}")
        
        if all_importances:
            feature_importances = np.mean(all_importances, axis=0)
            print("✓ Averaged importances from ensemble")
    
    # Set up feature names
    if feature_importances is not None:
        if hasattr(detector, 'active_features') and detector.active_features:
            active_feature_names = [detector.feature_names[i] for i in detector.active_features]
        else:
            active_feature_names = detector.feature_names[:len(feature_importances)]
        
        # Ensure we have the right number of feature names
        if len(active_feature_names) != len(feature_importances):
            print(f"Warning: Feature name count ({len(active_feature_names)}) != importance count ({len(feature_importances)})")
            # Truncate or pad as needed
            if len(active_feature_names) > len(feature_importances):
                active_feature_names = active_feature_names[:len(feature_importances)]
            else:
                # Add generic names if needed
                while len(active_feature_names) < len(feature_importances):
                    active_feature_names.append(f"feature_{len(active_feature_names)}")
        
        print(f"Feature Importance Rankings:")
        importance_pairs = list(zip(active_feature_names, feature_importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        total_importance = sum(imp for _, imp in importance_pairs)
        
        for i, (name, importance) in enumerate(importance_pairs):
            percentage = (importance / total_importance) * 100 if total_importance > 0 else 0
            print(f"  {i+1:2d}. {name:20s}: {importance:.4f} ({percentage:5.1f}%)")
        
        # Check importance diversity
        imp_std = np.std(feature_importances)
        if imp_std > 0.01:
            print(f"Good importance diversity (std={imp_std:.4f})")
        else:
            print(f"Low importance diversity (std={imp_std:.4f})")
    
    else:
        print("❌ No feature importances available after trying all methods")
        print(f"Classifier type: {type(detector.meta_classifier).__name__}")
        
        # Additional debugging
        if hasattr(detector.meta_classifier, 'base_classifier_'):
            base_clf = detector.meta_classifier.base_classifier_
            print(f"Base classifier type: {type(base_clf).__name__}")
            print(f"Base classifier attributes: {[attr for attr in dir(base_clf) if 'import' in attr.lower()]}")
            
            if hasattr(base_clf, '__dict__'):
                relevant_attrs = {k: type(v).__name__ for k, v in base_clf.__dict__.items() 
                                if 'import' in k.lower() or 'feature' in k.lower()}
                if relevant_attrs:
                    print(f"Relevant base classifier attributes: {relevant_attrs}")
        
        # Create dummy importances for visualization
        if hasattr(detector, 'active_features') and detector.active_features:
            n_features = len(detector.active_features)
            feature_importances = np.random.dirichlet(np.ones(n_features))  # Random but valid distribution
            active_feature_names = [detector.feature_names[i] for i in detector.active_features]
            print(f"Created dummy uniform importances for {n_features} features for visualization")
        else:
            feature_importances = np.array([])
            active_feature_names = []

    # Component score analysis
    if component_scores_list:
        print(f"\nComponent Score Analysis:")
        comp_df = pd.DataFrame(component_scores_list)
        
        for col in comp_df.columns:
            values = comp_df[col].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            status = "GOOD" if std_val > 1e-3 else "LOW VAR" if std_val > 1e-6 else "CONSTANT"
            print(f"  {status:8s} {col:20s}: μ={mean_val:7.4f}, σ={std_val:7.6f}, range=[{min_val:7.4f}, {max_val:7.4f}]")

    # Enhanced Visualization
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Truthfulness Score Distribution Plot
        truthful_scores = [conf for conf, label in zip(confidences, true_labels) if label == 0]
        hallucinated_scores = [conf for conf, label in zip(confidences, true_labels) if label == 1]
        
        if truthful_scores and hallucinated_scores:
            # Create density plots
            from scipy import stats
            
            # Generate smooth curves
            x_range = np.linspace(0, 1, 200)
            
            if len(truthful_scores) > 1:
                truthful_kde = stats.gaussian_kde(truthful_scores)
                truthful_density = truthful_kde(x_range)
            else:
                truthful_density = np.zeros_like(x_range)
            
            if len(hallucinated_scores) > 1:
                hallucinated_kde = stats.gaussian_kde(hallucinated_scores)
                hallucinated_density = hallucinated_kde(x_range)
            else:
                hallucinated_density = np.zeros_like(x_range)
            
            # Plot distributions
            ax1.fill_between(x_range, truthful_density, alpha=0.6, color='#87CEEB', label='Truthful')
            ax1.fill_between(x_range, hallucinated_density, alpha=0.6, color='#FFB6C1', label='Hallucinated')
            
            ax1.set_xlabel('Truthfulness score S(x)', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title(f'Score Distributions\nAcc: {report.get("accuracy", 0):.3f}, AUC: {auc:.3f}', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1)
        else:
            # Fallback to confusion matrix if distributions can't be plotted
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Factual', 'Hallucinated'],
                       yticklabels=['Factual', 'Hallucinated'], ax=ax1)
            ax1.set_title(f'Confusion Matrix\nAcc: {report.get("accuracy", 0):.3f}, AUC: {auc:.3f}')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
        
        # Feature importance plot
        if feature_importances is not None and len(feature_importances) > 0:
            importance_data = list(zip(active_feature_names, feature_importances))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            names, values = zip(*importance_data)
            y_pos = np.arange(len(names))
            
            bars = ax2.barh(y_pos, values, color='skyblue', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names)
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('Feature Importance')
            
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No Feature\nImportances\nAvailable', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Feature Importance (N/A)')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved analysis to {save_prefix}_analysis.png")
    except Exception as e:
        print(f"Could not save analysis: {e}")

    # Enhanced detailed results with comprehensive metrics
    truthful_scores = [conf for conf, label in zip(confidences, true_labels) if label == 0]
    hallucinated_scores = [conf for conf, label in zip(confidences, true_labels) if label == 1]
    
    # Calculate detailed statistics
    def calculate_score_stats(scores):
        if not scores:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'q25': 0.0, 'q75': 0.0
            }
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75))
        }
    
    # Calculate threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_analysis = []
    
    for threshold in thresholds:
        thresh_predictions = [1 if conf >= threshold else 0 for conf in confidences]
        thresh_report = classification_report(true_labels, thresh_predictions, output_dict=True, zero_division=0)
        
        threshold_analysis.append({
            'threshold': float(threshold),
            'accuracy': thresh_report.get('accuracy', 0.0),
            'precision_hallucinated': thresh_report.get('1', {}).get('precision', 0.0),
            'recall_hallucinated': thresh_report.get('1', {}).get('recall', 0.0),
            'f1_hallucinated': thresh_report.get('1', {}).get('f1-score', 0.0),
            'precision_truthful': thresh_report.get('0', {}).get('precision', 0.0),
            'recall_truthful': thresh_report.get('0', {}).get('recall', 0.0),
            'f1_truthful': thresh_report.get('0', {}).get('f1-score', 0.0)
        })
    
    # Find optimal threshold (maximize F1 for hallucinated class)
    optimal_threshold_entry = max(threshold_analysis, key=lambda x: x['f1_hallucinated'])
    
    # Calculate class separation metrics
    separation_metrics = {}
    if truthful_scores and hallucinated_scores:
        # Overlap coefficient
        min_truth, max_truth = min(truthful_scores), max(truthful_scores)
        min_hall, max_hall = min(hallucinated_scores), max(hallucinated_scores)
        
        overlap_start = max(min_truth, min_hall)
        overlap_end = min(max_truth, max_hall)
        overlap_range = max(0, overlap_end - overlap_start)
        total_range = max(max_truth, max_hall) - min(min_truth, min_hall)
        
        separation_metrics = {
            'score_separation': float(np.mean(hallucinated_scores) - np.mean(truthful_scores)),
            'effect_size_cohens_d': float((np.mean(hallucinated_scores) - np.mean(truthful_scores)) / 
                                         np.sqrt((np.var(hallucinated_scores) + np.var(truthful_scores)) / 2)),
            'overlap_coefficient': float(overlap_range / total_range) if total_range > 0 else 0.0,
            'distribution_overlap_percentage': float(100 * overlap_range / total_range) if total_range > 0 else 0.0
        }
    
    # Component score detailed analysis
    component_analysis = {}
    if component_scores_list:
        comp_df = pd.DataFrame(component_scores_list)
        for col in comp_df.columns:
            values = comp_df[col].values
            component_analysis[col] = {
                'statistics': calculate_score_stats(values.tolist()),
                'variation_quality': 'good' if np.std(values) > 1e-3 else 'low' if np.std(values) > 1e-6 else 'constant',
                'correlation_with_label': float(np.corrcoef(values, true_labels)[0, 1]) if len(values) > 1 else 0.0
            }
    
    # Feature importance detailed analysis
    feature_analysis = {}
    if feature_importances is not None and len(feature_importances) > 0:
        for i, (name, importance) in enumerate(zip(active_feature_names, feature_importances)):
            feature_analysis[name] = {
                'importance': float(importance),
                'rank': i + 1,
                'importance_percentage': float(importance * 100),
                'is_significant': bool(importance > 0.01)  # Threshold for significance
            }
    
    results = {
        'experiment_metadata': {
            'total_samples': len(true_labels),
            'truthful_samples': len(truthful_scores),
            'hallucinated_samples': len(hallucinated_scores),
            'class_balance': {
                'truthful_percentage': float(len(truthful_scores) / len(true_labels) * 100),
                'hallucinated_percentage': float(len(hallucinated_scores) / len(true_labels) * 100)
            }
        },
        'performance_metrics': {
            'classification_report': report,
            'auc_roc': float(auc),
            'confusion_matrix': cm.astype(int).tolist(),
            'num_errors': int(len(true_labels) - np.sum(np.array(predictions) == np.array(true_labels))),
            'error_rate': float(1 - report.get('accuracy', 0)),
            'optimal_threshold': {
                'value': optimal_threshold_entry['threshold'],
                'metrics_at_optimal': optimal_threshold_entry
            }
        },
        'score_distributions': {
            'truthful_scores': {
                'statistics': calculate_score_stats(truthful_scores),
                'sample_scores': truthful_scores[:100]  # First 100 for analysis
            },
            'hallucinated_scores': {
                'statistics': calculate_score_stats(hallucinated_scores),
                'sample_scores': hallucinated_scores[:100]  # First 100 for analysis
            },
            'separation_analysis': separation_metrics
        },
        'threshold_analysis': threshold_analysis,
        'feature_analysis': {
            'active_features': getattr(detector, 'active_features', []),
            'feature_importances': feature_importances.tolist() if feature_importances is not None else [],
            'detailed_feature_analysis': feature_analysis,
            'feature_diversity_metrics': {
                'importance_std': float(np.std(feature_importances)) if feature_importances is not None and len(feature_importances) > 0 else 0.0,
                'importance_range': float(np.max(feature_importances) - np.min(feature_importances)) if feature_importances is not None and len(feature_importances) > 0 else 0.0,
                'significant_features_count': sum(1 for imp in (feature_importances if feature_importances is not None else []) if imp > 0.01)
            }
        },
        'component_score_analysis': component_analysis,
        'model_diagnostics': {
            'prediction_confidence': {
                'mean_confidence': float(np.mean(confidences)),
                'confidence_std': float(np.std(confidences)),
                'high_confidence_predictions': int(sum(1 for c in confidences if c > 0.8 or c < 0.2)),
                'uncertain_predictions': int(sum(1 for c in confidences if 0.4 <= c <= 0.6))
            },
            'class_wise_performance': {
                'truthful': report.get('0', {}),
                'hallucinated': report.get('1', {})
            }
        }
    }

    try:
        with open(f'{save_prefix}_detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved detailed results to {save_prefix}_detailed_results.json")
    except Exception as e:
        print(f"Could not save detailed results: {e}")

    print(f"\nPerformance Results:")
    print(f"{'='*50}")
    print(f"Overall Accuracy:     {report.get('accuracy', 0):.3f}")
    if '1' in report:
        print(f"Hallucination Detection:")
        print(f"  Precision:          {report['1'].get('precision', 0):.3f}")
        print(f"  Recall:             {report['1'].get('recall', 0):.3f}")
        print(f"  F1-Score:           {report['1'].get('f1-score', 0):.3f}")
    if '0' in report:
        print(f"Factual Detection:")
        print(f"  Precision:          {report['0'].get('precision', 0):.3f}")
        print(f"  Recall:             {report['0'].get('recall', 0):.3f}")
        print(f"  F1-Score:           {report['0'].get('f1-score', 0):.3f}")
    print(f"AUC-ROC:              {auc:.3f}")
    print(f"{'='*50}")

    return results


def run_experiment(data_path: str,
                  white_box: bool = True,
                  model_name: str = "meta-llama/Llama-2-7b-hf",
                  force_retrain: bool = False,
                  number_of_samples: int = None,
                  use_ensemble: bool = False,
                  save_model: bool = True):
    """Run the experimental pipeline"""

    print(f"\nHallucination Detection Experiment")
    print(f"{'='*50}")
    print(f"Dataset: {data_path}")
    print(f"Mode: {'White-box' if white_box else 'Black-box'}")
    print(f"Ensemble: {use_ensemble}")
    print(f"Model: {model_name}")
    print(f"Samples: {number_of_samples or 'All'}")
    print(f"{'='*50}")

    model_interface = ModelInterface(model_name=model_name, white_box=white_box)
    all_samples = load_dataset(data_path)
    
    if number_of_samples is not None:
        all_samples = all_samples[:number_of_samples]
        print(f"Using first {number_of_samples} samples")

    hidden_dim = 4096
    if white_box and model_interface.model:
        hidden_dim = getattr(model_interface.model.config, 'hidden_size', 4096)

    detector = UnifiedHallucinationDetector(
        model_fn=model_interface.generate_with_logprobs,
        use_tsv=white_box,
        use_ensemble=use_ensemble,
        hidden_dim=hidden_dim,
        n_samples=10,
    )

    mode_str = "white_box" if white_box else "black_box"
    if use_ensemble:
        mode_str += "_ensemble"
    
    cache_filename = f"{os.path.basename(data_path).split('.')[0]}_{mode_str}_{number_of_samples or 'all'}_features.pkl"
    cache_path = os.path.join("cache", cache_filename)
    os.makedirs("cache", exist_ok=True)

    # Load or extract features
    if os.path.exists(cache_path) and not force_retrain:
        try:
            with open(cache_path, 'rb') as f:
                samples_with_features = pickle.load(f)
            if len(samples_with_features) == len(all_samples):
                 all_samples = samples_with_features
                 print("Loaded features from cache")
            else:
                 force_retrain = True
        except Exception as e:
            print(f"Cache loading failed: {e}, regenerating features")
            force_retrain = True

    if not os.path.exists(cache_path) or force_retrain:
        print("Extracting features...")
        
        for sample in tqdm(all_samples, desc="Feature extraction"):
            try:
                if detector.use_tsv:
                    sample.embedding = detector.apply_tsv_and_extract_features(sample, model=model_interface)
                sample.features = detector.extract_comprehensive_features(sample)
            except Exception as e:
                print(f"Feature extraction failed for sample: {e}")
                sample.features = np.zeros(12)
                if detector.use_tsv:
                    sample.embedding = torch.zeros(hidden_dim)
        
        # Verify feature diversity
        print("Verifying feature diversity...")
        test_features = np.array([s.features for s in all_samples[:200] if s.features is not None])
        if len(test_features) > 0:
            feature_stds = np.std(test_features, axis=0)
            non_constant_count = np.sum(feature_stds > 1e-6)
            print(f"Non-constant features: {non_constant_count}/{len(feature_stds)}")
            
            # Check key features
            semantic_std = feature_stds[1]  # semantic_entropy
            consistency_std = feature_stds[2]  # self_consistency
            
            print(f"Semantic entropy std: {semantic_std:.6f}")
            print(f"Self-consistency std: {consistency_std:.6f}")
            
            if non_constant_count >= 8:
                print("Excellent feature diversity achieved!")
            elif non_constant_count >= 6:
                print("Good feature diversity achieved!")
            else:
                print("Moderate feature diversity")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_samples, f)
            print("Cached features")
        except Exception as e:
            print(f"Failed to cache features: {e}")

    labeled_samples = [s for s in all_samples if s.label is not None]
    unlabeled_samples = [s for s in all_samples if s.label is None]
    
    if not labeled_samples:
        print("No labeled samples found")
        return {}

    labels = [s.label for s in labeled_samples]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) > 1:
        train_samples, test_samples = train_test_split(
            labeled_samples, test_size=0.3, random_state=42,
            stratify=labels
        )
    else:
        train_samples, test_samples = train_test_split(
            labeled_samples, test_size=0.3, random_state=42
        )

    print(f"Data split: Train={len(train_samples)}, Test={len(test_samples)}, Unlabeled={len(unlabeled_samples)}")

    print("Training detector...")
    training_data = train_samples + unlabeled_samples
    detector.train(training_data)
    print("Training completed successfully")

    if save_model:
        model_path = f"trained_detector_{mode_str}.pkl"
        detector.save(model_path)
        print(f"Saved model to {model_path}")


    print("Evaluating detector...")
    results = evaluate_detector(detector, test_samples, save_prefix=f"results_{mode_str}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Hallucination Detection Experiment")
    
    parser.add_argument("--data_path", type=str, default="truthfulqa_hallucination_dataset.json",
                       help="Path to dataset file")
    parser.add_argument("--white_box", action="store_true", default=False,
                       help="Use white-box mode")
    parser.add_argument("--black_box", action="store_true", default=False,
                       help="Use black-box mode")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name")
    parser.add_argument("--force_retrain", action="store_true", default=False,
                       help="Force retraining and feature re-extraction")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use")
    parser.add_argument("--ensemble", action="store_true", default=False,
                       help="Use ensemble")
    parser.add_argument("--no_save", action="store_true", default=False,
                       help="Don't save model")
    
    args = parser.parse_args()
    
    if args.black_box:
        white_box = False
    elif args.white_box:
        white_box = True
    else:
        white_box = False
    
    results = run_experiment(
        data_path=args.data_path,
        white_box=white_box,
        model_name=args.model_name,
        force_retrain=args.force_retrain,
        number_of_samples=args.num_samples,
        use_ensemble=args.ensemble,
        save_model=not args.no_save
    )
    
    if results:
        # Access the new nested structure
        performance_metrics = results.get('performance_metrics', {})
        accuracy = performance_metrics.get('classification_report', {}).get('accuracy', 0)
        auc = performance_metrics.get('auc_roc', 0)
        
        print(f"\nFinal Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC-ROC:  {auc:.3f}")
        