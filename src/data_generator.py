"""
data_generator.py - Generate training data for hidden face prediction

Each sample:
    - Input: 5 visible faces (45 stickers)
    - Target: 1 hidden face (9 stickers)
    - Hidden face index: which face is hidden (0-5)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from cube import RubiksCube


def generate_dataset(
    num_cubes: int,
    scramble_moves: int = 20,
    samples_per_cube: int = 6,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate dataset of (visible_faces, hidden_face) pairs.
    
    Args:
        num_cubes: Number of unique cube scrambles to generate
        scramble_moves: How many moves to scramble each cube
        samples_per_cube: How many samples per cube (1-6)
                        6 = hide each face once
                        1 = hide one random face
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
            'visible': (N, 5, 9) - the 5 visible faces
            'target': (N, 9) - the hidden face to predict
            'hidden_idx': (N,) - which face is hidden (0-5)
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_samples = num_cubes * samples_per_cube
    
    # Pre-allocate arrays
    visible = np.zeros((total_samples, 5, 9), dtype=np.int8)
    target = np.zeros((total_samples, 9), dtype=np.int8)
    hidden_idx = np.zeros(total_samples, dtype=np.int8)
    
    sample_i = 0
    
    for cube_i in range(num_cubes):
        # Create and scramble cube
        cube = RubiksCube()
        cube.scramble(scramble_moves)
        
        # Get all faces as (6, 9)
        all_faces = cube.get_all_faces()
        
        # Decide which faces to hide
        if samples_per_cube == 6:
            faces_to_hide = list(range(6))
        else:
            faces_to_hide = list(np.random.choice(6, samples_per_cube, replace=False))
        
        # Create samples
        for hide in faces_to_hide:
            # Visible faces: all except the hidden one
            visible_indices = [i for i in range(6) if i != hide]
            
            visible[sample_i] = all_faces[visible_indices]
            target[sample_i] = all_faces[hide]
            hidden_idx[sample_i] = hide
            
            sample_i += 1
        
        # Progress
        if (cube_i + 1) % 2000 == 0:
            print(f"  {cube_i + 1}/{num_cubes} cubes...")
    
    return {
        'visible': visible,
        'target': target,
        'hidden_idx': hidden_idx
    }


def split_dataset(
    data: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data: Dataset dictionary
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
                   Test gets the remaining (0.1)
        seed: Random seed
    
    Returns:
        (train_data, val_data, test_data) dictionaries
    """
    np.random.seed(seed)
    
    n = len(data['target'])
    indices = np.random.permutation(n)
    
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    def subset(idx):
        return {key: arr[idx] for key, arr in data.items()}
    
    return subset(train_idx), subset(val_idx), subset(test_idx)


def save_data(data: Dict[str, np.ndarray], filepath: str) -> None:
    """Save dataset to .npz file."""
    np.savez_compressed(filepath, **data)


def load_data(filepath: str) -> Dict[str, np.ndarray]:
    """Load dataset from .npz file."""
    loaded = np.load(filepath)
    return {key: loaded[key] for key in loaded.files}


def print_stats(data: Dict[str, np.ndarray], name: str = "Dataset") -> None:
    """Print dataset statistics."""
    n = len(data['target'])
    
    print(f"\n{name}:")
    print(f"  Samples: {n}")
    print(f"  Visible shape: {data['visible'].shape}")
    print(f"  Target shape: {data['target'].shape}")
    
    # Hidden face distribution
    counts = np.bincount(data['hidden_idx'], minlength=6)
    faces = ['U', 'D', 'F', 'B', 'L', 'R']
    dist = ', '.join(f"{f}:{c}" for f, c in zip(faces, counts))
    print(f"  Hidden face distribution: {dist}")
    
    # Memory
    mem_mb = sum(arr.nbytes for arr in data.values()) / 1024 / 1024
    print(f"  Memory: {mem_mb:.2f} MB")


# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Configuration
    NUM_CUBES = 10000        # 10k cubes
    SAMPLES_PER_CUBE = 6     # Hide each face once = 60k total samples
    SCRAMBLE_MOVES = 20
    SEED = 42
    
    print("="*50)
    print("GENERATING RUBIK'S CUBE DATASET")
    print("="*50)
    print(f"Cubes: {NUM_CUBES}")
    print(f"Samples per cube: {SAMPLES_PER_CUBE}")
    print(f"Total samples: {NUM_CUBES * SAMPLES_PER_CUBE}")
    print(f"Scramble moves: {SCRAMBLE_MOVES}")
    print()
    
    # Generate
    print("Generating data...")
    data = generate_dataset(
        num_cubes=NUM_CUBES,
        scramble_moves=SCRAMBLE_MOVES,
        samples_per_cube=SAMPLES_PER_CUBE,
        seed=SEED
    )
    print_stats(data, "Full dataset")
    
    # Split
    print("\nSplitting into train/val/test...")
    train, val, test = split_dataset(data)
    
    print_stats(train, "Train")
    print_stats(val, "Val")
    print_stats(test, "Test")
    
    # Save
    print("\nSaving...")
    save_data(train, str(data_dir / "train.npz"))
    save_data(val, str(data_dir / "val.npz"))
    save_data(test, str(data_dir / "test.npz"))
    
    print(f"\nSaved to: {data_dir}")
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)