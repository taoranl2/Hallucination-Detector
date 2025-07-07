# FILE: generate_truthfulqa_dataset.py
"""
Generate a test dataset based on TruthfulQA for hallucination detection
Compatible with unified_experiments.py
"""

import json
import random
import argparse
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def load_truthfulqa():
    """Load TruthfulQA dataset from Hugging Face"""
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation")
    return dataset['validation']  # TruthfulQA only has validation split


def create_hallucination_samples(
    truthfulqa_data,
    n_samples: Optional[int] = None,
    labeled_ratio: float = 0.8,
    include_both_types: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Create hallucination detection samples from TruthfulQA
    
    Args:
        truthfulqa_data: TruthfulQA dataset
        n_samples: Number of samples to generate (None = use all)
        labeled_ratio: Ratio of labeled vs unlabeled samples
        include_both_types: If True, include both truthful and hallucinated responses per question
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    samples = []
    
    # Determine number of samples
    total_questions = len(truthfulqa_data)
    if n_samples is None:
        n_samples = total_questions * (2 if include_both_types else 1)
    
    questions_to_use = min(total_questions, n_samples // (2 if include_both_types else 1))
    
    print(f"Processing {questions_to_use} questions...")
    
    for idx in tqdm(range(questions_to_use)):
        item = truthfulqa_data[idx]
        
        question = item['question'].strip()
        
        # Get best (truthful) answer
        best_answer = item['best_answer'].strip()
        
        # Get incorrect answers (hallucinated)
        incorrect_answers = item['incorrect_answers']
        
        # Create truthful sample
        if include_both_types or random.random() < 0.5:
            truthful_sample = {
                'query': question,
                'response': best_answer,
                'label': 0 if random.random() < labeled_ratio else None,  # 0 = factual
                'metadata': {
                    'source': 'truthfulqa',
                    'type': 'best_answer',
                    'question_id': idx
                }
            }
            samples.append(truthful_sample)
        
        # Create hallucinated sample
        if include_both_types or random.random() >= 0.5:
            if incorrect_answers:
                # Pick a random incorrect answer
                hallucinated_response = random.choice(incorrect_answers).strip()
                
                hallucinated_sample = {
                    'query': question,
                    'response': hallucinated_response,
                    'label': 1 if random.random() < labeled_ratio else None,  # 1 = hallucinated
                    'metadata': {
                        'source': 'truthfulqa',
                        'type': 'incorrect_answer',
                        'question_id': idx
                    }
                }
                samples.append(hallucinated_sample)
    
    # Shuffle samples
    random.shuffle(samples)
    
    return samples


def add_model_generated_responses(
    samples: List[Dict],
    model_name: str = "gpt2",
    n_generated: int = 100,
    labeled_ratio: float = 0.5
) -> List[Dict]:
    """
    Optionally add model-generated responses for more realistic testing
    This simulates responses from an actual LLM
    """
    print(f"\nNote: To add model-generated responses, you would need to:")
    print(f"1. Load a model (e.g., {model_name})")
    print(f"2. Generate responses for {n_generated} questions")
    print(f"3. Manually label or use heuristics to determine if they're hallucinated")
    print("Skipping this step for now (requires model access)\n")
    
    # In a real implementation, you would:
    # 1. Load the model
    # 2. Generate responses for random questions
    # 3. Add them with appropriate labels or as unlabeled
    
    return samples


def analyze_dataset(samples: List[Dict]) -> None:
    """Print dataset statistics"""
    total = len(samples)
    labeled = sum(1 for s in samples if s['label'] is not None)
    unlabeled = total - labeled
    
    factual = sum(1 for s in samples if s['label'] == 0)
    hallucinated = sum(1 for s in samples if s['label'] == 1)
    
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {total}")
    print(f"Labeled samples: {labeled} ({labeled/total*100:.1f}%)")
    print(f"  - Factual: {factual} ({factual/labeled*100:.1f}% of labeled)")
    print(f"  - Hallucinated: {hallucinated} ({hallucinated/labeled*100:.1f}% of labeled)")
    print(f"Unlabeled samples: {unlabeled} ({unlabeled/total*100:.1f}%)")
    
    # Sample questions
    print("\n=== Sample Questions ===")
    for i in range(min(3, len(samples))):
        s = samples[i]
        label_str = "Factual" if s['label'] == 0 else "Hallucinated" if s['label'] == 1 else "Unlabeled"
        print(f"\nQuestion {i+1} ({label_str}):")
        print(f"Q: {s['query'][:100]}...")
        print(f"A: {s['response'][:100]}...")


def save_dataset(samples: List[Dict], output_path: str, 
                 include_metadata: bool = False) -> None:
    """Save dataset to JSON file"""
    
    # Remove metadata if not needed (to match expected format)
    if not include_metadata:
        clean_samples = []
        for s in samples:
            clean_sample = {
                'query': s['query'],
                'response': s['response'],
                'label': s['label']
            }
            clean_samples.append(clean_sample)
        samples = clean_samples
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to: {output_path}")


def create_specialized_subsets(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Create specialized subsets for different experiments"""
    
    subsets = {
        'full': samples,
        'labeled_only': [s for s in samples if s['label'] is not None],
        'balanced': [],
        'few_shot': []
    }
    
    # Create balanced subset
    factual = [s for s in samples if s['label'] == 0]
    hallucinated = [s for s in samples if s['label'] == 1]
    min_size = min(len(factual), len(hallucinated))
    
    subsets['balanced'] = (
        random.sample(factual, min_size) + 
        random.sample(hallucinated, min_size)
    )
    random.shuffle(subsets['balanced'])
    
    # Create few-shot subset (small labeled + many unlabeled)
    n_few_shot = 50  # 50 labeled examples
    labeled = [s for s in samples if s['label'] is not None]
    unlabeled = [s for s in samples if s['label'] is None]
    
    if len(labeled) >= n_few_shot:
        few_shot_labeled = random.sample(labeled, n_few_shot)
        subsets['few_shot'] = few_shot_labeled + unlabeled
        random.shuffle(subsets['few_shot'])
    
    return subsets


def main():
    parser = argparse.ArgumentParser(
        description="Generate TruthfulQA-based dataset for hallucination detection"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='truthfulqa_hallucination_dataset.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--n_samples', '-n',
        type=int,
        default=None,
        help='Number of samples to generate (None = use all)'
    )
    parser.add_argument(
        '--labeled_ratio', '-l',
        type=float,
        default=0.8,
        help='Ratio of labeled samples (default: 0.8)'
    )
    parser.add_argument(
        '--include_metadata', '-m',
        action='store_true',
        help='Include metadata in output'
    )
    parser.add_argument(
        '--create_subsets', '-s',
        action='store_true',
        help='Create specialized subsets'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load TruthfulQA
    truthfulqa = load_truthfulqa()
    
    # Create samples
    samples = create_hallucination_samples(
        truthfulqa,
        n_samples=args.n_samples,
        labeled_ratio=args.labeled_ratio,
        include_both_types=True,
        seed=args.seed
    )
    
    # Analyze dataset
    analyze_dataset(samples)
    
    # Save main dataset
    save_dataset(samples, args.output, args.include_metadata)
    
    # Create and save subsets if requested
    if args.create_subsets:
        subsets = create_specialized_subsets(samples)
        
        for name, subset in subsets.items():
            if subset and name != 'full':
                subset_path = args.output.replace('.json', f'_{name}.json')
                save_dataset(subset, subset_path, args.include_metadata)
                print(f"Subset '{name}' saved with {len(subset)} samples")


if __name__ == "__main__":
    main()