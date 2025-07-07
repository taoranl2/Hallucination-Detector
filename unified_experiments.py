# FILE: unified_experiments.py
"""
Experimental framework for the unified hallucination detection system

NOTE: If you encounter gradient errors with white_box=True, try running with 
white_box=False first (black-box mode) which doesn't require model access.
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
from unified_hallucination_detector import UnifiedHallucinationDetector, HallucinationSample
from tqdm import tqdm  # Add tqdm import
import warnings
import os
import hashlib
warnings.filterwarnings('ignore')


class ModelInterface:
    """Interface for LLM interaction supporting both white-box and black-box access"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 white_box: bool = True):
        
        self.device = device
        self.white_box = white_box
        
        if white_box:
            print(f"Loading model {model_name} in white-box mode...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Set the model's pad_token_id to match the tokenizer to avoid warnings
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            print("Running in black-box mode (API simulation)")
            self.tokenizer = None
            self.model = None
            
    def generate_with_logprobs(self, prompt: str, temperature: float = 0.9, 
                              max_length: int = 100) -> Tuple[str, float]:
        """Generate response with log probability"""
        if self.white_box:
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
                
            response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Calculate average log probability
            if outputs.scores:
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    token_id = outputs.sequences[0, i + inputs.input_ids.shape[1]]
                    log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                    log_probs.append(log_prob)
                avg_log_prob = np.mean(log_probs) if log_probs else -10.0
            else:
                avg_log_prob = -10.0
        else:
            # Black-box simulation
            response = f"Simulated response to: {prompt[:50]}..."
            avg_log_prob = np.random.uniform(-5, -0.1)
            
        return response, avg_log_prob
    
    def get_hidden_states(self, prompt: str, response: str, layer: int = -1) -> torch.Tensor:
        """Get hidden states from specific layer (white-box only)"""
        if not self.white_box:
            # Return mock hidden states for black-box
            seq_len = len((prompt + " " + response).split())
            return torch.randn(seq_len, 4096)
            
        full_text = prompt + " " + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
        return hidden_states[0]  # Remove batch dimension


def load_dataset(filepath: str) -> List[HallucinationSample]:
    """Load dataset from JSON or CSV"""
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif filepath.endswith('.csv'):
        data = pd.read_csv(filepath).to_dict('records')
    else:
        raise ValueError("Unsupported file format")
    
    samples = []
    for item in data:
        sample = HallucinationSample(
            query=item['query'],
            response=item['response'],
            label=item.get('label', None)
        )
        samples.append(sample)
        
    return samples

def evaluate_detector(detector: UnifiedHallucinationDetector,
                     test_samples: List[HallucinationSample],
                     save_prefix: str = "results") -> Dict:
    """Comprehensive evaluation of the detector"""
    
    predictions = []
    confidences = []
    all_scores = []
    true_labels = []

    # Use tqdm for progress bar instead of print statements
    for sample in tqdm(test_samples, desc="Evaluating"):
        if sample.label is None:
            continue
            
        pred, conf, scores = detector.predict(sample.query, sample.response)
        predictions.append(pred)
        confidences.append(conf)
        all_scores.append(scores)
        true_labels.append(sample.label)
    
    # Calculate metrics
    report = classification_report(true_labels, predictions, output_dict=True)
    auc = roc_auc_score(true_labels, confidences)
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{save_prefix}_confusion_matrix.png')
    plt.close()
    
    # Plot score distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract component scores
    mi_scores = [s['mi_score'] for s in all_scores]
    entropy_scores = [s['semantic_entropy'] for s in all_scores]
    consistency_scores = [s['self_consistency'] for s in all_scores]
    tsv_scores = [s['tsv_score'] for s in all_scores]
    
    # Separate by true label
    factual_idx = [i for i, l in enumerate(true_labels) if l == 0]
    hall_idx = [i for i, l in enumerate(true_labels) if l == 1]
    
    # Plot distributions
    for ax, scores, title in zip(axes.flat, 
                                 [mi_scores, entropy_scores, consistency_scores, tsv_scores],
                                 ['MI Score', 'Semantic Entropy', 'Self Consistency', 'TSV Score']):
        ax.hist([scores[i] for i in factual_idx], bins=20, alpha=0.7, label='Factual', density=True)
        ax.hist([scores[i] for i in hall_idx], bins=20, alpha=0.7, label='Hallucinated', density=True)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{title} Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_score_distributions.png')
    plt.close()
    
    # Error analysis
    errors = []
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != true:
            errors.append({
                'index': i,
                'query': test_samples[i].query,
                'response': test_samples[i].response,
                'predicted': int(pred),  # Convert to int
                'true': int(true),  # Convert to int
                'confidence': float(confidences[i]),  # Convert to float
                'scores': {k: float(v) for k, v in all_scores[i].items()}  # Convert all scores to float
            })
    
    # Save detailed results - ensure all numpy types are converted
    results = {
        'classification_report': report,
        'auc_roc': float(auc),  # Convert to float
        'confusion_matrix': cm.astype(int).tolist(),  # Convert to int then list
        'num_errors': len(errors),
        'error_rate': float(len(errors) / len(true_labels)),  # Convert to float
        'top_errors': errors[:10]  # Save top 10 errors
    }
    
    with open(f'{save_prefix}_detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Precision: {report['1']['precision']:.3f}")
    print(f"Recall: {report['1']['recall']:.3f}")
    print(f"F1-Score: {report['1']['f1-score']:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Error Rate: {results['error_rate']:.3f}")
    
    return results


def ablation_study(samples: List[HallucinationSample], 
                  model_interface: ModelInterface) -> Dict:
    """Perform ablation study on different components"""
    
    # Split data
    labeled_samples = [s for s in samples if s.label is not None]
    train_samples, test_samples = train_test_split(labeled_samples, test_size=0.3, random_state=42)
    unlabeled_samples = [s for s in samples if s.label is None]
    
    # Combine train and unlabeled
    all_train_samples = train_samples + unlabeled_samples
    
    configurations = [
        ("Full Model", {"use_tsv": True, "n_samples": 10}),
        ("No TSV", {"use_tsv": False, "n_samples": 10}),
        ("Few Samples", {"use_tsv": True, "n_samples": 3}),
        ("No Unlabeled", {"use_tsv": True, "n_samples": 10})
    ]
    
    results = {}
    
    for name, config in configurations:
        print(f"\n=== Testing {name} ===")
        
        # Create detector with configuration
        detector = UnifiedHallucinationDetector(
            model_fn=model_interface.generate_with_logprobs,
            **config
        )
        
        # Use appropriate training data
        if name == "No Unlabeled":
            detector.train(train_samples)
        else:
            detector.train(all_train_samples)
        
        # Evaluate
        eval_results = evaluate_detector(detector, test_samples, save_prefix=f"ablation_{name.replace(' ', '_')}")
        results[name] = eval_results
        
    # Compare results
    print("\n=== Ablation Study Results ===")
    print(f"{'Configuration':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 50)
    
    for name, res in results.items():
        acc = res['classification_report']['accuracy']
        f1 = res['classification_report']['1']['f1-score']
        auc = res['auc_roc']
        print(f"{name:<20} {acc:<10.3f} {f1:<10.3f} {auc:<10.3f}")
    
    return results

def generate_model_filename(data_path: str, white_box: bool, model_name: str, 
                          n_samples: int = 2, n_epochs: int = 5, number_of_samples: int = None) -> str:
    """Generate a unique filename for the trained model based on parameters"""
    # Extract dataset name
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    
    # Sanitize model name for filename
    model_name_clean = model_name.replace('/', '_').replace('-', '_')
    
    # Create parameter string
    if number_of_samples is not None:   
        params = f"{dataset_name}_wb{white_box}_model{model_name_clean}_ns{n_samples}_ep{n_epochs}_n{number_of_samples}"
    else:
        params = f"{dataset_name}_wb{white_box}_model{model_name_clean}_ns{n_samples}_ep{n_epochs}_all"
    
    # Create models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    return f"saved_models/{params}.pkl"

def run_full_experiment(data_path: str,
                       white_box: bool = True,
                       model_name: str = "meta-llama/Llama-2-7b-hf",
                       force_retrain: bool = False,
                       number_of_samples: int = None):
    """Run complete experimental pipeline
    
    Args:
        data_path: Path to JSON dataset file (required)
        white_box: Whether to use white-box mode with TSV
        model_name: Name of the model to use
        force_retrain: Force retraining even if saved model exists
    """
    
    if white_box:
        print("=" * 70)
        print("Running in WHITE-BOX mode with TSV")
        print("This requires a real language model with accessible hidden states.")
        print("If you encounter gradient errors, try white_box=False instead.")
        print("=" * 70)
    else:
        print("=" * 70)
        print("Running in BLACK-BOX mode (no TSV, API-only)")
        print("This mode works without model access and is good for testing.")
        print("=" * 70)
    
    # Initialize model interface
    model_interface = ModelInterface(model_name=model_name, white_box=white_box)
    
    # Load dataset (required parameter now)
    print(f"\nLoading dataset from: {data_path}")
    samples = load_dataset(data_path)
    if number_of_samples is not None:
        samples = samples[:number_of_samples]
    # Split data
    labeled_samples = [s for s in samples if s.label is not None]
    train_samples, test_samples = train_test_split(
        labeled_samples, test_size=0.3, random_state=42, 
        stratify=[s.label for s in labeled_samples]
    )
    unlabeled_samples = [s for s in samples if s.label is None]
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(samples)}")
    print(f"Labeled samples: {len(labeled_samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Unlabeled samples: {len(unlabeled_samples)}")
    
    # Generate model filename
    model_filename = generate_model_filename(
        data_path=data_path,
        white_box=white_box,
        model_name=model_name,
        n_samples=5,  # Match the n_samples in UnifiedHallucinationDetector
        n_epochs=5,    # Match the epochs in train method
        number_of_samples=number_of_samples
    )
    
    # Check if model exists and load if not forcing retrain
    if os.path.exists(model_filename) and not force_retrain:
        print(f"\nFound existing model: {model_filename}")
        print("Loading trained model...")
        detector = UnifiedHallucinationDetector.load(
            model_filename, 
            model_fn=model_interface.generate_with_logprobs
        )
    else:
        # Initialize unified detector
        print("\nNo existing model found or force_retrain=True")
        print("Training new model...")
        detector = UnifiedHallucinationDetector(
            model_fn=model_interface.generate_with_logprobs,
            use_tsv=white_box,  # Only use TSV in white-box mode
            hidden_dim=4096 if "llama" in model_name.lower() else 4096
        )
        
        # Train
        print("\nTraining unified detector...")
        all_train_samples = train_samples + unlabeled_samples
        detector.train(all_train_samples)
        
        # Save the trained model
        detector.save(model_filename)
    
    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = evaluate_detector(detector, test_samples)
    
    # # Ablation study
    # if white_box and force_retrain:  # Only do ablation when training new model
    #     print("\nPerforming ablation study...")
    #     ablation_results = ablation_study(samples, model_interface)
    
    # # Interactive testing
    # print("\n=== Interactive Testing ===")
    # print("Enter 'quit' to exit")
    
    # while True:
    #     query = input("\nEnter query: ")
    #     if query.lower() == 'quit':
    #         break
            
    #     response = input("Enter response: ")
        
    #     prediction, confidence, scores = detector.predict(query, response)
        
    #     print(f"\nPrediction: {'HALLUCINATED' if prediction == 1 else 'FACTUAL'}")
    #     print(f"Confidence: {confidence:.3f}")
    #     print("\nComponent Scores:")
    #     for name, value in scores.items():
    #         print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    # For real data with white-box access:
    run_full_experiment(
        data_path="truthfulqa_hallucination_dataset.json", 
        white_box=True, 
        model_name="meta-llama/Meta-Llama-3-8B",
        force_retrain=False,
        number_of_samples=1000
    )
    
    # For real data with black-box access:
    # run_full_experiment(
    #     data_path="truthfulqa_hallucination_dataset.json", 
    #     white_box=False,
    #     model_name="meta-llama/Meta-Llama-3-8B"
    # )
