"""
verify_data.py - Quick check that generated data is correct
"""

import numpy as np
from pathlib import Path
from cube import RubiksCube


def verify_sample(visible: np.ndarray, target: np.ndarray, hidden_idx: int) -> bool:
    """
    Verify a single sample has valid color counts.
    
    A valid cube has exactly 9 stickers of each color (0-5).
    """
    # Reconstruct full cube
    all_faces = np.zeros((6, 9), dtype=np.int8)
    
    visible_indices = [i for i in range(6) if i != hidden_idx]
    for i, vi in enumerate(visible_indices):
        all_faces[vi] = visible[i]
    all_faces[hidden_idx] = target
    
    # Check color counts
    flat = all_faces.flatten()
    for color in range(6):
        if np.sum(flat == color) != 9:
            return False
    return True


def main():
    data_dir = Path(__file__).parent.parent / "data"
    
    print("Loading test data...")
    data = np.load(data_dir / "test.npz")
    visible = data['visible']
    target = data['target']
    hidden_idx = data['hidden_idx']
    
    print(f"Samples: {len(target)}")
    
    # Verify random samples
    print("\nVerifying 100 random samples...")
    indices = np.random.choice(len(target), 100, replace=False)
    
    valid_count = 0
    for i in indices:
        if verify_sample(visible[i], target[i], hidden_idx[i]):
            valid_count += 1
    
    print(f"Valid: {valid_count}/100")
    
    # Show one example
    print("\n" + "="*50)
    print("EXAMPLE SAMPLE")
    print("="*50)
    
    idx = 0
    face_names = ['U', 'D', 'F', 'B', 'L', 'R']
    hidden = hidden_idx[idx]
    visible_names = [f for i, f in enumerate(face_names) if i != hidden]
    
    print(f"\nHidden face: {face_names[hidden]} (index {hidden})")
    print(f"\nVisible faces ({visible_names}):")
    print(visible[idx])
    print(f"\nTarget (face {face_names[hidden]}):")
    print(target[idx])
    
    # Verify this specific sample
    if verify_sample(visible[idx], target[idx], hidden):
        print("\n✓ Sample is valid!")
    else:
        print("\n✗ Sample is invalid!")


if __name__ == "__main__":
    main()