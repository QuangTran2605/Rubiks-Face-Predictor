"""
Generate training data for the Rubik's cube predictor.

Each sample:
- Input: 5 visible faces
- Output: 1 hidden face
- The hidden face is randomly chosen (U, D, F, B, L, or R)
"""

import json
import os
import random
from cube import RubiksCube, FULL_COLORS


def generate_samples(num_samples, num_scramble_moves=20):
    """
    Generate random cube samples.
    
    Args:
        num_samples: How many samples to generate
        num_scramble_moves: How many moves to scramble each cube
        
    Returns:
        List of samples
    """
    samples = []
    all_faces = ['U', 'D', 'F', 'B', 'L', 'R']
    
    for i in range(num_samples):
        # Create and scramble a cube
        cube = RubiksCube()
        cube.scramble(num_scramble_moves)
        
        # Randomly choose which face is hidden
        hidden_face_name = random.choice(all_faces)
        
        # Get visible and hidden faces
        visible = cube.get_visible_faces(hidden=hidden_face_name)
        hidden = cube.get_hidden_face(hidden=hidden_face_name)
        
        # Store as a sample
        sample = {
            'visible': visible,
            'hidden_face_name': hidden_face_name,
            'hidden': hidden
        }
        samples.append(sample)
        
        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} / {num_samples} samples")
    
    return samples


def save_data(samples, filename):
    """Save samples to a JSON file."""
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    
    with open(filepath, 'w') as f:
        json.dump(samples, f)
    
    print(f"Saved {len(samples)} samples to {filepath}")


def load_data(filename):
    """Load samples from a JSON file."""
    filepath = os.path.join('data', filename)
    
    with open(filepath, 'r') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} samples from {filepath}")
    return samples


def print_sample(sample):
    """Print a sample nicely."""
    print(f"\nHidden face: {sample['hidden_face_name']}")
    
    print("\nVisible faces (INPUT):")
    for face_name, face in sample['visible'].items():
        colors = []
        for row in face:
            row_colors = [FULL_COLORS[c] for c in row]
            colors.append(row_colors)
        print(f"  {face_name}: {colors[0]}")
        print(f"      {colors[1]}")
        print(f"      {colors[2]}")
    
    print("\nHidden face (OUTPUT to predict):")
    hidden = sample['hidden']
    for row in hidden:
        row_colors = [FULL_COLORS[c] for c in row]
        print(f"  {row_colors}")


if __name__ == "__main__":
    # Generate training data
    print("=== Generating Training Data ===")
    print("This may take a moment...\n")
    train_samples = generate_samples(100000)
    save_data(train_samples, 'train.json')
    
    # Generate validation data
    print("\n=== Generating Validation Data ===")
    val_samples = generate_samples(20000)
    save_data(val_samples, 'val.json')
    
    # Show example samples
    print("\n=== Example Samples ===")
    
    # Show samples with different hidden faces
    shown_faces = set()
    for sample in train_samples:
        if sample['hidden_face_name'] not in shown_faces:
            print_sample(sample)
            shown_faces.add(sample['hidden_face_name'])
            print("\n" + "="*50)
        
        if len(shown_faces) >= 3:  # Show 3 examples
            break
    
    # Statistics
    print("\n=== Dataset Statistics ===")
    
    # Count hidden face distribution
    hidden_counts = {'U': 0, 'D': 0, 'F': 0, 'B': 0, 'L': 0, 'R': 0}
    for sample in train_samples:
        hidden_counts[sample['hidden_face_name']] += 1
    
    print("\nHidden face distribution (should be roughly equal):")
    for face, count in hidden_counts.items():
        percentage = count / len(train_samples) * 100
        print(f"  {face}: {count} ({percentage:.2f}%)")