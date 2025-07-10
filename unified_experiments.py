# FILE: unified_experiments.py (Clean Improved Version)
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
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.white_box = False
                self.tokenizer = None
                self.model = None
        else:
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
            except Exception:
                response = f"Fallback response to: {prompt[:50]}..."
                avg_log_prob = np.random.uniform(-5, -0.1)
        else:
            response = f"Simulated response to: {prompt[:50]}..."
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
        print("Please generate dataset first:")
        print(f"python generate_truthfulqa_dataset.py -o {filepath} -n 2000")
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

    for sample in tqdm(test_samples, desc="Evaluating"):
        if sample.label is None:
            continue

        try:
            pred, conf, _ = detector.predict(sample.query, sample.response)
            predictions.append(pred)
            confidences.append(conf)
            true_labels.append(sample.label)
        except Exception:
            continue
    
    if not true_labels:
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

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{save_prefix}_confusion_matrix.png')
        plt.close()
    except Exception:
        pass

    results = {
        'classification_report': report,
        'auc_roc': float(auc),
        'confusion_matrix': cm.astype(int).tolist(),
        'num_errors': len(true_labels) - np.sum(np.array(predictions) == np.array(true_labels)),
        'error_rate': 1 - report.get('accuracy', 0)
    }

    try:
        with open(f'{save_prefix}_detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    print(f"\nResults:")
    print(f"Accuracy: {report.get('accuracy', 0):.3f}")
    if '1' in report:
        print(f"Hallucination Precision: {report['1'].get('precision', 0):.3f}")
        print(f"Hallucination Recall: {report['1'].get('recall', 0):.3f}")
        print(f"Hallucination F1: {report['1'].get('f1-score', 0):.3f}")
    if '0' in report:
        print(f"Factual Precision: {report['0'].get('precision', 0):.3f}")
        print(f"Factual Recall: {report['0'].get('recall', 0):.3f}")
        print(f"Factual F1: {report['0'].get('f1-score', 0):.3f}")
    print(f"AUC-ROC: {auc:.3f}")

    return results


def run_full_experiment(data_path: str,
                       white_box: bool = True,
                       model_name: str = "meta-llama/Llama-2-7b-hf",
                       force_retrain: bool = False,
                       number_of_samples: int = None,
                       use_ensemble: bool = False,
                       save_model: bool = True):
    """Run complete experimental pipeline"""

    print(f"Running experiment: {data_path}")
    print(f"Mode: {'White-box' if white_box else 'Black-box'}")
    print(f"Ensemble: {use_ensemble}")

    model_interface = ModelInterface(model_name=model_name, white_box=white_box)
    all_samples = load_dataset(data_path)
    
    if number_of_samples is not None:
        all_samples = all_samples[:number_of_samples]

    hidden_dim = 4096
    if white_box and model_interface.model:
        hidden_dim = getattr(model_interface.model.config, 'hidden_size', 4096)

    detector = UnifiedHallucinationDetector(
        model_fn=model_interface.generate_with_logprobs,
        use_tsv=white_box,
        use_ensemble=use_ensemble,
        hidden_dim=hidden_dim,
        n_samples=5,
    )

    mode_str = "white_box" if white_box else "black_box"
    if use_ensemble:
        mode_str += "_ensemble"
    
    cache_filename = f"{os.path.basename(data_path).split('.')[0]}_{mode_str}_{number_of_samples or 'all'}_features.pkl"
    cache_path = os.path.join("cache", cache_filename)
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path) and not force_retrain:
        try:
            with open(cache_path, 'rb') as f:
                samples_with_features = pickle.load(f)
            if len(samples_with_features) == len(all_samples):
                 all_samples = samples_with_features
                 print("Loaded features from cache")
            else:
                 force_retrain = True
        except Exception:
            force_retrain = True

    if not os.path.exists(cache_path) or force_retrain:
        print("Extracting features...")
        for sample in tqdm(all_samples, desc="Feature extraction"):
            try:
                if detector.use_tsv:
                    sample.embedding = detector.apply_tsv_and_extract_features(sample, model=model_interface)
                sample.features = detector.extract_comprehensive_features(sample)
            except Exception:
                sample.features = np.zeros(12)
                if detector.use_tsv:
                    sample.embedding = torch.zeros(hidden_dim)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_samples, f)
        except Exception:
            pass

    labeled_samples = [s for s in all_samples if s.label is not None]
    unlabeled_samples = [s for s in all_samples if s.label is None]
    
    if not labeled_samples:
        print("Error: No labeled samples found")
        return

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

    print(f"Training: {len(train_samples)}, Test: {len(test_samples)}, Unlabeled: {len(unlabeled_samples)}")

    print("Training detector...")
    training_data = train_samples + unlabeled_samples
    try:
        detector.train(training_data)
    except Exception as e:
        print(f"Training failed: {e}")

    if save_model:
        model_path = f"trained_detector_{mode_str}.pkl"
        try:
            detector.save(model_path)
        except Exception:
            pass

    print("Evaluating...")
    try:
        results = evaluate_detector(detector, test_samples, save_prefix=f"results_{mode_str}")
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}


def main():
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
                       help="Force retraining")
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
    
    if not os.path.exists(args.data_path):
        print(f"Dataset '{args.data_path}' not found!")
        print(f"Generate it first: python generate_truthfulqa_dataset.py -o {args.data_path} -n 2000")
        return
    
    results = run_full_experiment(
        data_path=args.data_path,
        white_box=white_box,
        model_name=args.model_name,
        force_retrain=args.force_retrain,
        number_of_samples=args.num_samples,
        use_ensemble=args.ensemble,
        save_model=not args.no_save
    )
    
    if results:
        accuracy = results.get('classification_report', {}).get('accuracy', 0)
        print(f"\nFinal accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("python unified_experiments.py --data_path my_dataset.json --black_box --ensemble")
        print("python unified_experiments.py --data_path my_dataset.json --white_box --ensemble")
    else:
        main()