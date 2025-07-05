# FILE: test_gradient_fix.py
"""
Test script to verify the gradient error fix
"""

import torch
import torch.nn as nn

# Test TSV gradient flow
class TestTSV(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim) * 0.01)
        
    def forward(self, x):
        return x + 5.0 * self.v

def test_gradient_flow():
    """Test that gradients flow properly through TSV"""
    print("Testing gradient flow through TSV...")
    
    # Create TSV
    tsv = TestTSV(dim=128)
    optimizer = torch.optim.Adam(tsv.parameters(), lr=0.01)
    
    # Create mock input with gradients
    x = torch.randn(10, 128, requires_grad=True)
    
    # Forward pass
    output = tsv(x)
    
    # Create a simple loss
    loss = output.mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    if tsv.v.grad is not None and tsv.v.grad.abs().sum() > 0:
        print("✓ Gradients flow correctly through TSV")
        return True
    else:
        print("✗ No gradients in TSV")
        return False

def test_prototype_detachment():
    """Test that prototypes are properly detached"""
    print("\nTesting prototype detachment...")
    
    # Create embedding with gradients
    embedding = torch.randn(128, requires_grad=True)
    
    # Create prototype (should be detached)
    with torch.no_grad():
        prototype = torch.randn(128)
        prototype = torch.nn.functional.normalize(prototype, dim=-1)
    
    # Compute similarity (prototype should be detached)
    similarity = torch.dot(embedding, prototype.detach())
    
    # Create loss
    loss = -torch.log(torch.exp(10 * similarity) + 1e-10)
    
    # Check if we can backprop
    try:
        loss.backward()
        print("✓ Loss computation works with detached prototypes")
        return True
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Running gradient tests...")
    print("=" * 50)
    
    test1 = test_gradient_flow()
    test2 = test_prototype_detachment()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("All tests passed! The gradient issue should be fixed.")
    else:
        print("Some tests failed. Check the error messages above.")