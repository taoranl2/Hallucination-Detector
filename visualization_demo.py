# FILE: visualization_demo.py
"""
Visualization and demonstration of the unified hallucination detection system
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from typing import List, Dict
import pandas as pd


def visualize_latent_space(embeddings_before: np.ndarray, 
                          embeddings_after: np.ndarray,
                          labels: List[int],
                          save_path: str = "latent_space_visualization.png"):
    """
    Visualize the effect of TSV on latent space separation
    Similar to Figure 1 in the TSV paper
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # T-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    
    # Before TSV
    embedded_before = tsne.fit_transform(embeddings_before)
    colors = ['blue' if l == 0 else 'red' for l in labels]
    ax1.scatter(embedded_before[:, 0], embedded_before[:, 1], c=colors, alpha=0.6)
    ax1.set_title("Pre-trained Embeddings", fontsize=14)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    
    # After TSV
    embedded_after = tsne.fit_transform(embeddings_after)
    ax2.scatter(embedded_after[:, 0], embedded_after[:, 1], c=colors, alpha=0.6)
    ax2.set_title("Steered Embeddings by TSV", fontsize=14)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Truthful'),
                      Patch(facecolor='red', label='Hallucinated')]
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latent space visualization saved to {save_path}")


def plot_mi_iterative_prompting(query: str, 
                               responses: List[str],
                               mi_scores: List[float],
                               save_path: str = "mi_iterative_prompting.png"):
    """
    Visualize MI scores from iterative prompting
    Following the approach in "To Believe or Not to Believe Your LLM"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create iterative prompts
    prompts = [query]
    for i, resp in enumerate(responses[:-1]):
        prompts.append(f"{query}\nPrevious: {resp}")
    
    # Plot MI scores
    x = range(len(mi_scores))
    ax.plot(x, mi_scores, 'b-o', linewidth=2, markersize=8)
    
    # Add threshold line
    threshold = np.mean(mi_scores) + np.std(mi_scores)
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    
    # Annotations
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("MI Score", fontsize=12)
    ax.set_title(f"Mutual Information with Iterative Prompting\nQuery: {query[:50]}...", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MI iterative prompting visualization saved to {save_path}")


def plot_component_contributions(scores_dict: Dict[str, List[float]],
                               labels: List[int],
                               save_path: str = "component_contributions.png"):
    """
    Visualize how different components contribute to detection
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    components = ['mi_score', 'semantic_entropy', 'self_consistency', 'tsv_score']
    titles = ['MI-based Uncertainty', 'Semantic Entropy', 'Self-Consistency', 'TSV Score']
    
    for ax, comp, title in zip(axes, components, titles):
        if comp in scores_dict:
            scores = scores_dict[comp]
            
            # Separate by label
            factual_scores = [s for s, l in zip(scores, labels) if l == 0]
            hall_scores = [s for s, l in zip(scores, labels) if l == 1]
            
            # Plot distributions
            ax.hist(factual_scores, bins=20, alpha=0.7, label='Factual', 
                   color='blue', density=True)
            ax.hist(hall_scores, bins=20, alpha=0.7, label='Hallucinated', 
                   color='red', density=True)
            
            ax.set_xlabel(title, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add separation score
            if len(factual_scores) > 0 and len(hall_scores) > 0:
                sep_score = abs(np.mean(hall_scores) - np.mean(factual_scores)) / \
                           (np.std(hall_scores) + np.std(factual_scores) + 1e-6)
                ax.text(0.05, 0.95, f'Separation: {sep_score:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Component Score Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Component contributions visualization saved to {save_path}")


def plot_performance_comparison(results: Dict[str, Dict[str, float]],
                              save_path: str = "performance_comparison.png"):
    """
    Compare performance across different methods and configurations
    """
    # Extract data
    methods = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'auc-roc']
    
    # Create data matrix
    data = []
    for method in methods:
        row = []
        for metric in metrics:
            if metric == 'auc-roc':
                value = results[method].get('auc_roc', 0)
            else:
                value = results[method].get('classification_report', {}).get('1', {}).get(metric, 
                         results[method].get('classification_report', {}).get(metric, 0))
            row.append(value)
        data.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(data, index=methods, columns=[m.title() for m in metrics])
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax)
    
    ax.set_title("Performance Comparison Across Methods", fontsize=16)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Methods", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison saved to {save_path}")


def create_demo_visualizations():
    """
    Create all demonstration visualizations
    """
    # Simulate data for visualization
    np.random.seed(42)
    n_samples = 200
    
    # 1. Latent space visualization (before/after TSV)
    # Before: overlapping distributions
    embeddings_before = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n_samples//2),
        np.random.multivariate_normal([1, 1], [[1, 0.8], [0.8, 1]], n_samples//2)
    ])
    
    # After: well-separated distributions
    embeddings_after = np.vstack([
        np.random.multivariate_normal([-2, 0], [[0.5, 0.1], [0.1, 0.5]], n_samples//2),
        np.random.multivariate_normal([2, 0], [[0.5, 0.1], [0.1, 0.5]], n_samples//2)
    ])
    
    labels = [0] * (n_samples//2) + [1] * (n_samples//2)
    
    visualize_latent_space(embeddings_before, embeddings_after, labels)
    
    # 2. MI iterative prompting
    query = "What is the capital of France?"
    responses = ["Paris", "London", "Berlin", "Madrid"]
    mi_scores = [0.2, 0.5, 1.2, 1.8]  # Increasing MI indicates uncertainty
    
    plot_mi_iterative_prompting(query, responses, mi_scores)
    
    # 3. Component contributions
    scores_dict = {
        'mi_score': np.concatenate([
            np.random.normal(0.3, 0.2, n_samples//2),
            np.random.normal(0.8, 0.3, n_samples//2)
        ]),
        'semantic_entropy': np.concatenate([
            np.random.normal(0.4, 0.15, n_samples//2),
            np.random.normal(0.7, 0.2, n_samples//2)
        ]),
        'self_consistency': np.concatenate([
            np.random.normal(0.8, 0.1, n_samples//2),
            np.random.normal(0.3, 0.2, n_samples//2)
        ]),
        'tsv_score': np.concatenate([
            np.random.normal(0.7, 0.15, n_samples//2),
            np.random.normal(-0.2, 0.3, n_samples//2)
        ])
    }
    
    plot_component_contributions(scores_dict, labels)
    
    # 4. Performance comparison
    results = {
        'Unified System': {
            'classification_report': {
                'accuracy': 0.892,
                '1': {'precision': 0.88, 'recall': 0.85, 'f1-score': 0.865}
            },
            'auc_roc': 0.912
        },
        'TSV Only': {
            'classification_report': {
                'accuracy': 0.82,
                '1': {'precision': 0.80, 'recall': 0.78, 'f1-score': 0.79}
            },
            'auc_roc': 0.84
        },
        'MI Only': {
            'classification_report': {
                'accuracy': 0.76,
                '1': {'precision': 0.74, 'recall': 0.72, 'f1-score': 0.73}
            },
            'auc_roc': 0.78
        },
        'Semantic Entropy': {
            'classification_report': {
                'accuracy': 0.71,
                '1': {'precision': 0.69, 'recall': 0.68, 'f1-score': 0.685}
            },
            'auc_roc': 0.73
        }
    }
    
    plot_performance_comparison(results)
    
    print("\nAll visualizations created successfully!")
    print("Files created:")
    print("- latent_space_visualization.png")
    print("- mi_iterative_prompting.png")
    print("- component_contributions.png")
    print("- performance_comparison.png")


if __name__ == "__main__":
    create_demo_visualizations()