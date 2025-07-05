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
import warnings
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
                    output_scores=True
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


def create_synthetic_dataset(n_samples: int = 1000, 
                           labeled_ratio: float = 0.2) -> List[HallucinationSample]:
    """Create synthetic dataset for testing"""
    np.random.seed(42)
    
    # Templates and entities
    templates = [
        ("What is the capital of {}?", ["France", "Germany", "Japan", "Mars", "Atlantis"]),
        ("Who wrote {}?", ["1984", "Harry Potter", "The Bible", "The Internet", "Python"]),
        ("When was {} invented?", ["the wheel", "electricity", "time travel", "unicorns"]),
        ("What is the formula for {}?", ["water", "gold", "happiness", "dark matter"]),
        ("Who discovered {}?", ["America", "gravity", "fire", "the meaning of life"])
    ]
    
    samples = []
    
    for i in range(n_samples):
        template, entities = templates[i % len(templates)]
        entity = np.random.choice(entities)
        query = template.format(entity)
        
        # Generate response and label
        if "Mars" in entity or "Atlantis" in entity or "time travel" in entity or \
           "unicorns" in entity or "happiness" in entity or "dark matter" in entity or \
           "meaning of life" in entity:
            # Hallucinated
            response = f"The answer is definitely {entity} which everyone knows"
            true_label = 1
        else:
            # Factual
            response = f"The answer is related to {entity} based on factual information"
            true_label = 0
            
        # Assign label based on ratio
        if i < int(n_samples * labeled_ratio):
            label = true_label
        else:
            label = None  # Unlabeled
            
        samples.append(HallucinationSample(
            query=query,
            response=response,
            label=label
        ))
        
    return samples


def evaluate_detector(detector: UnifiedHallucinationDetector,
                     test_samples: List[HallucinationSample],
                     save_prefix: str = "results") -> Dict:
    """Comprehensive evaluation of the detector"""
    
    predictions = []
    confidences = []
    all_scores = []
    true_labels = []
    
    print("Evaluating on test set...")
    for i, sample in enumerate(test_samples):
        if sample.label is None:
            continue
            
        pred, conf, scores = detector.predict(sample.query, sample.response)
        predictions.append(pred)
        confidences.append(conf)
        all_scores.append(scores)
        true_labels.append(sample.label)
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(test_samples)} samples")
    
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
                'predicted': pred,
                'true': true,
                'confidence': confidences[i],
                'scores': all_scores[i]
            })
    
    # Save detailed results
    results = {
        'classification_report': report,
        'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'num_errors': len(errors),
        'error_rate': len(errors) / len(true_labels),
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


def run_full_experiment(data_path: Optional[str] = None,
                       white_box: bool = True,
                       model_name: str = "meta-llama/Llama-2-7b-hf"):
    """Run complete experimental pipeline"""
    
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
    
    # Load or create dataset
    if data_path:
        samples = load_dataset(data_path)
    else:
        print("Creating synthetic dataset...")
        samples = create_synthetic_dataset(n_samples=1000, labeled_ratio=0.2)
    
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
    
    # Initialize unified detector
    detector = UnifiedHallucinationDetector(
        model_fn=model_interface.generate_with_logprobs,
        use_tsv=white_box,  # Only use TSV in white-box mode
        hidden_dim=4096 if "llama" in model_name.lower() else 4096
    )
    
    # Train
    print("\nTraining unified detector...")
    all_train_samples = train_samples + unlabeled_samples
    detector.train(all_train_samples)
    
    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = evaluate_detector(detector, test_samples)
    
    # Ablation study
    if white_box:
        print("\nPerforming ablation study...")
        ablation_results = ablation_study(samples, model_interface)
    
    # Interactive testing
    print("\n=== Interactive Testing ===")
    print("Enter 'quit' to exit")
    
    while True:
        query = input("\nEnter query: ")
        if query.lower() == 'quit':
            break
            
        response = input("Enter response: ")
        
        prediction, confidence, scores = detector.predict(query, response)
        
        print(f"\nPrediction: {'HALLUCINATED' if prediction == 1 else 'FACTUAL'}")
        print(f"Confidence: {confidence:.3f}")
        print("\nComponent Scores:")
        for name, value in scores.items():
            print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    # Example usage
    # For synthetic data with black-box access (safer for testing):
    print("Running in black-box mode first (no TSV, no model needed)...")
    run_full_experiment(white_box=False)
    
    # For synthetic data with white-box access:
    # run_full_experiment(white_box=True)
    
    # For real data with black-box access:
    # run_full_experiment(data_path="path/to/data.json", white_box=False)
    
    # For specific model:
    # run_full_experiment(white_box=True, model_name="meta-llama/Llama-3-8b")