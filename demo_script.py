# FILE: demo_script.py
"""
Simple demonstration script for the unified hallucination detection system
Run this to see the system in action with minimal setup
"""

import numpy as np
from unified_hallucination_detector import UnifiedHallucinationDetector, HallucinationSample
from visualization_demo import create_demo_visualizations
import warnings
warnings.filterwarnings('ignore')


def create_demo_samples():
    """Create a small set of demo samples"""
    samples = [
        # Labeled factual examples
        HallucinationSample(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            label=0
        ),
        HallucinationSample(
            query="Who wrote Romeo and Juliet?",
            response="Romeo and Juliet was written by William Shakespeare.",
            label=0
        ),
        HallucinationSample(
            query="What is 2+2?",
            response="2+2 equals 4.",
            label=0
        ),
        
        # Labeled hallucinated examples
        HallucinationSample(
            query="What is the capital of Mars?",
            response="The capital of Mars is New Olympus City.",
            label=1
        ),
        HallucinationSample(
            query="Who invented the internet in 1850?",
            response="The internet was invented by Charles Babbage in 1850.",
            label=1
        ),
        HallucinationSample(
            query="What color is the wind?",
            response="The wind is typically blue in color.",
            label=1
        ),
        
        # Unlabeled examples
        HallucinationSample(
            query="What is the largest planet?",
            response="Jupiter is the largest planet in our solar system.",
            label=None
        ),
        HallucinationSample(
            query="When was the telephone invented?",
            response="The telephone was invented in 1876 by Alexander Graham Bell.",
            label=None
        ),
        HallucinationSample(
            query="What is the speed of darkness?",
            response="The speed of darkness is exactly twice the speed of light.",
            label=None
        ),
        HallucinationSample(
            query="How many continents are there?",
            response="There are seven continents on Earth.",
            label=None
        ),
    ]
    
    return samples


def run_demo():
    """Run a simple demonstration of the system"""
    print("=" * 60)
    print("UNIFIED HALLUCINATION DETECTION SYSTEM - DEMO")
    print("=" * 60)
    
    # Create demo samples
    print("\n1. Creating demo dataset...")
    samples = create_demo_samples()
    print(f"   - Created {len(samples)} samples")
    print(f"   - Labeled: {sum(1 for s in samples if s.label is not None)}")
    print(f"   - Unlabeled: {sum(1 for s in samples if s.label is None)}")
    
    # Initialize detector (black-box mode for demo)
    print("\n2. Initializing detector in black-box mode...")
    detector = UnifiedHallucinationDetector(
        use_tsv=False,  # Black-box mode
        n_samples=5     # Fewer samples for faster demo
    )
    
    # Train the detector
    print("\n3. Training detector with semi-supervised learning...")
    detector.train(samples, n_initial_epochs=5, n_augmented_epochs=5, k_confident=3)
    
    # Test on new examples
    print("\n4. Testing on new examples:")
    print("-" * 60)
    
    test_cases = [
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the capital of Atlantis?", "The capital of Atlantis is Poseidon City."),
        ("How many days in a week?", "There are seven days in a week."),
        ("What year is it on Mars?", "On Mars, it's currently the year 2387 in the Martian calendar."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
        ("What is the chemical formula for water?", "The chemical formula for water is H2O2."),
    ]
    
    for query, response in test_cases:
        try:
            prediction, confidence, scores = detector.predict(query, response)
            
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            print(f"Prediction: {'HALLUCINATED' if prediction == 1 else 'FACTUAL'}")
            print(f"Confidence: {confidence:.3f}")
            print("Component scores:")
            for component, score in scores.items():
                print(f"  - {component}: {score:.3f}")
            print("-" * 60)
        except Exception as e:
            print(f"Error processing: {query}")
            print(f"Error: {str(e)}")
            continue
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    try:
        create_demo_visualizations()
        print("   Visualizations saved!")
    except Exception as e:
        print(f"   Could not create visualizations: {str(e)}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    
    # Summary
    print("\nSummary:")
    print("- The system successfully trained on minimal labeled data")
    print("- It leveraged unlabeled samples through semi-supervised learning")
    print("- Multiple detection methods were combined for robust performance")
    print("- The system can adapt to both white-box and black-box scenarios")
    
    print("\nNext steps:")
    print("1. Try with your own data (JSON format)")
    print("2. Enable white-box mode with a real LLM for better performance")
    print("3. Experiment with different configurations in ablation studies")
    print("4. Check the visualizations for insights into the detection process")


if __name__ == "__main__":
    run_demo()