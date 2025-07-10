# FILE: generate_multi_dataset.py
"""
Generate a test dataset using both TruthfulQA and TriviaQA for hallucination detection
Creates a much larger and more diverse dataset
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
    return dataset['validation']


def load_triviaqa():
    """Load TriviaQA dataset from Hugging Face"""
    print("Loading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "rc")
    # Use validation split, limit size for practical reasons
    return dataset['validation']


def create_truthfulqa_samples(truthfulqa_data, n_samples: Optional[int] = None, 
                             labeled_ratio: float = 0.8, seed: int = 42) -> List[Dict]:
    """Create samples from TruthfulQA (same as before)"""
    random.seed(seed)
    
    samples = []
    total_questions = len(truthfulqa_data)
    
    if n_samples is None:
        questions_to_use = total_questions
    else:
        questions_to_use = min(total_questions, n_samples // 2)  # Half from TruthfulQA
    
    print(f"Processing {questions_to_use} TruthfulQA questions...")
    
    for idx in tqdm(range(questions_to_use), desc="TruthfulQA"):
        item = truthfulqa_data[idx]
        question = item['question'].strip()
        best_answer = item['best_answer'].strip()
        incorrect_answers = item['incorrect_answers']
        
        # Create truthful sample
        truthful_sample = {
            'query': question,
            'response': best_answer,
            'label': 0 if random.random() < labeled_ratio else None,
            'metadata': {
                'source': 'truthfulqa',
                'type': 'best_answer',
                'question_id': idx
            }
        }
        samples.append(truthful_sample)
        
        # Create hallucinated sample
        if incorrect_answers:
            hallucinated_response = random.choice(incorrect_answers).strip()
            hallucinated_sample = {
                'query': question,
                'response': hallucinated_response,
                'label': 1 if random.random() < labeled_ratio else None,
                'metadata': {
                    'source': 'truthfulqa',
                    'type': 'incorrect_answer',
                    'question_id': idx
                }
            }
            samples.append(hallucinated_sample)
    
    return samples


def create_triviaqa_samples(triviaqa_data, n_samples: Optional[int] = None,
                           labeled_ratio: float = 0.8, seed: int = 42,
                           hallucination_templates: List[str] = None) -> List[Dict]:
    """Create samples from TriviaQA"""
    random.seed(seed + 1)  # Different seed to avoid correlation
    
    if hallucination_templates is None:
        hallucination_templates = [
            "Actually, the answer is {fake_answer}.",
            "The correct answer is {fake_answer}.",
            "It's {fake_answer}.",
            "The answer is definitely {fake_answer}.",
            "{fake_answer} is the right answer.",
            "Most people think it's {fake_answer}.",
            "According to recent research, it's {fake_answer}.",
            "The widely accepted answer is {fake_answer}."
        ]
    
    # Common wrong answers for different question types
    fake_answers = [
        "Paris", "London", "New York", "Tokyo", "Berlin", "Rome", "Madrid", "Amsterdam",
        "1969", "1945", "1776", "2001", "1989", "1492", "1865", "1914",
        "Shakespeare", "Einstein", "Napoleon", "Lincoln", "Washington", "Churchill", "Galileo",
        "Blue", "Red", "Green", "Yellow", "Purple", "Orange", "Black", "White",
        "100", "50", "25", "200", "500", "1000", "10", "5",
        "Carbon", "Oxygen", "Hydrogen", "Nitrogen", "Gold", "Silver", "Iron", "Copper",
        "Dog", "Cat", "Lion", "Tiger", "Eagle", "Shark", "Elephant", "Whale"
    ]
    
    samples = []
    total_questions = len(triviaqa_data)
    
    if n_samples is None:
        questions_to_use = min(total_questions, 2000)  # Limit TriviaQA to reasonable size
    else:
        questions_to_use = min(total_questions, n_samples // 2)  # Half from TriviaQA
    
    print(f"Processing {questions_to_use} TriviaQA questions...")
    
    for idx in tqdm(range(questions_to_use), desc="TriviaQA"):
        item = triviaqa_data[idx]
        question = item['question'].strip()
        
        # Get the answer (TriviaQA has multiple answer formats)
        answer_dict = item['answer']
        if 'value' in answer_dict:
            correct_answer = answer_dict['value'].strip()
        elif 'normalized_value' in answer_dict:
            correct_answer = answer_dict['normalized_value'].strip()
        else:
            # Skip if no clear answer
            continue
            
        if not correct_answer or len(correct_answer) < 2:
            continue
        
        # Create factual sample
        factual_sample = {
            'query': question,
            'response': correct_answer,
            'label': 0 if random.random() < labeled_ratio else None,
            'metadata': {
                'source': 'triviaqa',
                'type': 'correct_answer',
                'question_id': idx
            }
        }
        samples.append(factual_sample)
        
        # Create hallucinated sample
        # Choose a fake answer that's different from the correct one
        fake_answer = random.choice(fake_answers)
        while fake_answer.lower() in correct_answer.lower() or correct_answer.lower() in fake_answer.lower():
            fake_answer = random.choice(fake_answers)
        
        # Use template to create hallucinated response
        template = random.choice(hallucination_templates)
        hallucinated_response = template.format(fake_answer=fake_answer)
        
        hallucinated_sample = {
            'query': question,
            'response': hallucinated_response,
            'label': 1 if random.random() < labeled_ratio else None,
            'metadata': {
                'source': 'triviaqa',
                'type': 'fake_answer',
                'question_id': idx,
                'original_answer': correct_answer
            }
        }
        samples.append(hallucinated_sample)
    
    return samples


def create_cross_contamination_samples(truthfulqa_data, triviaqa_data, 
                                     n_samples: int = 100, labeled_ratio: float = 0.8,
                                     seed: int = 42) -> List[Dict]:
    """Create additional hallucinated samples by mixing answers between datasets"""
    random.seed(seed + 2)
    
    samples = []
    
    # Get some TruthfulQA questions and TriviaQA answers
    tqa_questions = [item['question'].strip() for item in truthfulqa_data[:50]]
    trivia_answers = []
    
    for item in triviaqa_data[:100]:
        answer_dict = item['answer']
        if 'value' in answer_dict:
            answer = answer_dict['value'].strip()
            if answer and len(answer) > 2:
                trivia_answers.append(answer)
    
    print(f"Creating {n_samples} cross-contamination samples...")
    
    for i in range(min(n_samples, len(tqa_questions))):
        question = random.choice(tqa_questions)
        wrong_answer = random.choice(trivia_answers)
        
        # Make it sound confident but wrong
        templates = [
            f"The answer is clearly {wrong_answer}.",
            f"Based on my knowledge, it's {wrong_answer}.",
            f"I'm confident the answer is {wrong_answer}.",
            f"The correct response is {wrong_answer}."
        ]
        
        hallucinated_response = random.choice(templates)
        
        sample = {
            'query': question,
            'response': hallucinated_response,
            'label': 1 if random.random() < labeled_ratio else None,
            'metadata': {
                'source': 'cross_contamination',
                'type': 'mixed_wrong_answer',
                'question_id': i
            }
        }
        samples.append(sample)
    
    return samples


def analyze_dataset(samples: List[Dict]) -> None:
    """Print dataset statistics"""
    total = len(samples)
    labeled = sum(1 for s in samples if s['label'] is not None)
    unlabeled = total - labeled
    
    factual = sum(1 for s in samples if s['label'] == 0)
    hallucinated = sum(1 for s in samples if s['label'] == 1)
    
    # Source breakdown
    sources = {}
    for s in samples:
        source = s.get('metadata', {}).get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print("\n=== Enhanced Dataset Statistics ===")
    print(f"Total samples: {total}")
    print(f"Labeled samples: {labeled} ({labeled/total*100:.1f}%)")
    print(f"  - Factual: {factual} ({factual/labeled*100:.1f}% of labeled)")
    print(f"  - Hallucinated: {hallucinated} ({hallucinated/labeled*100:.1f}% of labeled)")
    print(f"Unlabeled samples: {unlabeled} ({unlabeled/total*100:.1f}%)")
    print()
    print("Source breakdown:")
    for source, count in sources.items():
        print(f"  - {source}: {count} ({count/total*100:.1f}%)")
    
    # Sample questions by source
    print("\n=== Sample Questions by Source ===")
    for source in sources.keys():
        source_samples = [s for s in samples if s.get('metadata', {}).get('source') == source]
        if source_samples:
            sample = source_samples[0]
            label_str = "Factual" if sample['label'] == 0 else "Hallucinated" if sample['label'] == 1 else "Unlabeled"
            print(f"\n{source.upper()} example ({label_str}):")
            print(f"Q: {sample['query'][:100]}...")
            print(f"A: {sample['response'][:100]}...")


def save_dataset(samples: List[Dict], output_path: str, include_metadata: bool = False) -> None:
    """Save dataset to JSON file"""
    
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
    
    print(f"\nEnhanced dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced dataset using TruthfulQA + TriviaQA for hallucination detection"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='enhanced_hallucination_dataset.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--n_samples', '-n',
        type=int,
        default=None,
        help='Total number of samples to generate (None = use all available)'
    )
    parser.add_argument(
        '--labeled_ratio', '-l',
        type=float,
        default=0.8,
        help='Ratio of labeled samples (default: 0.8)'
    )
    parser.add_argument(
        '--truthfulqa_only',
        action='store_true',
        help='Use only TruthfulQA (skip TriviaQA)'
    )
    parser.add_argument(
        '--triviaqa_only',
        action='store_true',
        help='Use only TriviaQA (skip TruthfulQA)'
    )
    parser.add_argument(
        '--include_metadata', '-m',
        action='store_true',
        help='Include metadata in output'
    )
    parser.add_argument(
        '--no_cross_contamination',
        action='store_true',
        help='Skip cross-contamination samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    all_samples = []
    
    # Load and process TruthfulQA
    if not args.triviaqa_only:
        truthfulqa = load_truthfulqa()
        truthfulqa_samples = create_truthfulqa_samples(
            truthfulqa,
            n_samples=args.n_samples,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed
        )
        all_samples.extend(truthfulqa_samples)
        print(f"Added {len(truthfulqa_samples)} TruthfulQA samples")
    
    # Load and process TriviaQA
    if not args.truthfulqa_only:
        triviaqa = load_triviaqa()
        triviaqa_samples = create_triviaqa_samples(
            triviaqa,
            n_samples=args.n_samples,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed
        )
        all_samples.extend(triviaqa_samples)
        print(f"Added {len(triviaqa_samples)} TriviaQA samples")
    
    # Add cross-contamination samples
    if not args.no_cross_contamination and not args.truthfulqa_only and not args.triviaqa_only:
        cross_samples = create_cross_contamination_samples(
            truthfulqa, triviaqa,
            n_samples=100,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed
        )
        all_samples.extend(cross_samples)
        print(f"Added {len(cross_samples)} cross-contamination samples")
    
    # Shuffle all samples
    random.seed(args.seed)
    random.shuffle(all_samples)
    
    # Limit to requested number if specified
    if args.n_samples is not None and len(all_samples) > args.n_samples:
        all_samples = all_samples[:args.n_samples]
    
    # Analyze and save
    analyze_dataset(all_samples)
    save_dataset(all_samples, args.output, args.include_metadata)
    
    print(f"\n✅ Enhanced dataset generation complete!")
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"🎯 Ready for training with: python unified_experiments.py --data_path {args.output} --black_box --ensemble")


if __name__ == "__main__":
    main()