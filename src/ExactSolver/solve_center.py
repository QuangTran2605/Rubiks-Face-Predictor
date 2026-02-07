"""
solver_center.py - Solve center piece (trivial)
"""

import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / "data"

def solve_center(visible_faces: np.ndarray) -> int:
    """
    Find the missing center color.
    
    Args:
        visible_faces: (5, 9) array - 5 visible faces flattened
    
    Returns:
        Missing center color (0-5)
    """
    visible_centers = set()
    for face_data in visible_faces:
        visible_centers.add(int(face_data[4]))  # Position 4 is center
    
    all_colors = set(range(6))
    missing = all_colors - visible_centers
    
    return next(iter(missing))


def test():
    print("="*60)
    print("TESTING CENTER SOLVER")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / "train.npz")
    
    visible = data['visible']
    target = data['target']
    
    n = len(target)
    correct = sum(
        solve_center(visible[i]) == target[i][4]
        for i in range(n)
    )
    
    print(f"Samples: {n}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/n:.4f}")
    
    if correct == n:
        print("\n✓ CENTER SOLVER: PERFECT!")
    else:
        print("\n✗ CENTER SOLVER: HAS ERRORS")


if __name__ == "__main__":
    test()