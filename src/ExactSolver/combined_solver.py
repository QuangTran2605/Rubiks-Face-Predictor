"""
combined_solver.py - Combine center, corner, and edge solvers

Pipeline:
    solve_center → solve_corners → solve_edges → combine → validate

Returns all possible hidden face patterns.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from itertools import product

# Import individual solvers
from solve_center import solve_center
from solver_corners import solve_corners
from solver_edges import solve_edges

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / "data"

# =============================================================================
# CONSTANTS
# =============================================================================

U, D, F, B, L, R = 0, 1, 2, 3, 4, 5
FACE_NAMES = ['U', 'D', 'F', 'B', 'L', 'R']
COLOR_NAMES = ['W', 'Y', 'G', 'B', 'O', 'R']

# Hidden face sticker positions
CENTER_POS = 4
CORNER_POSITIONS = [0, 2, 6, 8]
EDGE_POSITIONS = [1, 3, 5, 7]

# =============================================================================
# COMBINED SOLVER
# =============================================================================

def solve_hidden_face(
    visible_faces: np.ndarray, 
    hidden_face_idx: int,
    return_components: bool = False
) -> List[np.ndarray]:
    """
    Solve for all possible hidden face patterns.
    
    Args:
        visible_faces: (5, 9) array - 5 visible faces flattened
        hidden_face_idx: which face is hidden (0-5)
        return_components: if True, return detailed breakdown
    
    Returns:
        List of possible hidden face patterns, each as (9,) array
        
        If return_components=True, returns tuple:
        (patterns, center, corner_solutions, edge_solutions)
    """
    # Step 1: Solve center (deterministic)
    center_color = solve_center(visible_faces)
    
    # Step 2: Solve corners (may have multiple solutions)
    corner_solutions = solve_corners(visible_faces, hidden_face_idx)
    
    if not corner_solutions:
        if return_components:
            return [], center_color, [], []
        return []
    
    # Step 3: Solve edges (may have multiple solutions)
    edge_solutions = solve_edges(visible_faces, hidden_face_idx)
    
    if not edge_solutions:
        if return_components:
            return [], center_color, corner_solutions, []
        return []
    
    # Step 4: Combine all solutions
    patterns = []
    
    for corner_config, edge_config in product(corner_solutions, edge_solutions):
        pattern = build_pattern(center_color, corner_config, edge_config)
        patterns.append(pattern)
    
    # Remove duplicates (if any)
    patterns = deduplicate_patterns(patterns)
    
    if return_components:
        return patterns, center_color, corner_solutions, edge_solutions
    
    return patterns


def build_pattern(
    center_color: int,
    corner_config: Dict[int, int],
    edge_config: Dict[int, int]
) -> np.ndarray:
    """
    Build a complete hidden face pattern from components.
    
    Args:
        center_color: color for position 4
        corner_config: {pos: color} for positions 0, 2, 6, 8
        edge_config: {pos: color} for positions 1, 3, 5, 7
    
    Returns:
        (9,) array representing full hidden face
    """
    pattern = np.zeros(9, dtype=np.int8)
    
    # Center
    pattern[CENTER_POS] = center_color
    
    # Corners
    for pos, color in corner_config.items():
        pattern[pos] = color
    
    # Edges
    for pos, color in edge_config.items():
        pattern[pos] = color
    
    return pattern


def deduplicate_patterns(patterns: List[np.ndarray]) -> List[np.ndarray]:
    """Remove duplicate patterns."""
    if not patterns:
        return []
    
    seen = set()
    unique = []
    
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    return unique


# =============================================================================
# VALIDATION (Future: add solvability check)
# =============================================================================

def validate_pattern(
    visible_faces: np.ndarray,
    hidden_face_idx: int,
    pattern: np.ndarray
) -> bool:
    """
    Validate that a pattern produces a solvable cube state.
    
    TODO: Implement parity checks
    - Edge orientation parity
    - Corner orientation parity  
    - Permutation parity
    
    For now, returns True (no validation).
    """
    # Placeholder for future solvability validation
    return True


def solve_hidden_face_validated(
    visible_faces: np.ndarray,
    hidden_face_idx: int
) -> List[np.ndarray]:
    """
    Solve and filter for valid (solvable) patterns only.
    """
    patterns = solve_hidden_face(visible_faces, hidden_face_idx)
    
    valid_patterns = [
        p for p in patterns
        if validate_pattern(visible_faces, hidden_face_idx, p)
    ]
    
    return valid_patterns


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pattern_to_string(pattern: np.ndarray) -> str:
    """Convert pattern to readable string."""
    colors = [COLOR_NAMES[c] for c in pattern]
    return f"""
    {colors[0]} {colors[1]} {colors[2]}
    {colors[3]} {colors[4]} {colors[5]}
    {colors[6]} {colors[7]} {colors[8]}
    """.strip()


def check_solution(
    patterns: List[np.ndarray],
    target: np.ndarray
) -> Tuple[bool, int]:
    """
    Check if target pattern is in solution set.
    
    Returns:
        (found, index) - found=True if target in patterns, index is position
    """
    for i, p in enumerate(patterns):
        if np.array_equal(p, target):
            return True, i
    return False, -1


# =============================================================================
# TEST
# =============================================================================

def test():
    print("=" * 60)
    print("TESTING COMBINED SOLVER")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / "train.npz")
    
    visible = data['visible']
    target = data['target']
    hidden_idx = data['hidden_idx']
    
    n = len(target)
    
    # Statistics
    total_samples = 0
    samples_with_solution = 0
    samples_correct = 0
    solution_count_dist = {}
    
    # Track worst cases (most solutions)
    max_solutions = 0
    max_solutions_idx = -1
    
    errors = []
    
    for i in range(n):
        total_samples += 1
        
        patterns = solve_hidden_face(visible[i], hidden_idx[i])
        
        count = len(patterns)
        solution_count_dist[count] = solution_count_dist.get(count, 0) + 1
        
        if count > max_solutions:
            max_solutions = count
            max_solutions_idx = i
        
        if count == 0:
            errors.append((i, "No solution"))
            continue
        
        samples_with_solution += 1
        
        # Check if target is in solutions
        found, idx = check_solution(patterns, target[i])
        
        if found:
            samples_correct += 1
        else:
            errors.append((i, f"Target not in {count} solutions"))
    
    # Print results
    print(f"\nResults:")
    print(f"  Total samples:         {total_samples}")
    print(f"  Samples with solution: {samples_with_solution}")
    print(f"  Samples correct:       {samples_correct}")
    print(f"  Accuracy:              {samples_correct / total_samples:.4f}")
    
    print(f"\nSolution count distribution:")
    for count in sorted(solution_count_dist.keys()):
        freq = solution_count_dist[count]
        pct = 100 * freq / total_samples
        bar = "█" * int(pct / 2)
        print(f"  {count:3d} solutions: {freq:5d} ({pct:5.1f}%) {bar}")
    
    # Statistics
    counts = []
    for count, freq in solution_count_dist.items():
        counts.extend([count] * freq)
    
    print(f"\nSolution count statistics:")
    print(f"  Min:    {min(counts)}")
    print(f"  Max:    {max(counts)}")
    print(f"  Mean:   {np.mean(counts):.2f}")
    print(f"  Median: {np.median(counts):.1f}")
    
    if samples_correct == total_samples:
        print("\n" + "=" * 60)
        print("✓ COMBINED SOLVER: PERFECT!")
        print("=" * 60)
    else:
        print(f"\n✗ COMBINED SOLVER: {total_samples - samples_correct} errors")
        
        print(f"\nFirst 5 errors:")
        for i, reason in errors[:5]:
            print(f"  Sample {i}: {reason}")
    
    # Show example with most solutions
    if max_solutions_idx >= 0:
        print(f"\n" + "=" * 60)
        print(f"EXAMPLE: Sample {max_solutions_idx} with {max_solutions} solutions")
        print("=" * 60)
        debug_sample(
            visible[max_solutions_idx],
            hidden_idx[max_solutions_idx],
            target[max_solutions_idx]
        )


def debug_sample(
    visible_faces: np.ndarray,
    hidden_face_idx: int,
    target: np.ndarray
):
    """Debug a single sample with detailed output."""
    print(f"\nHidden face: {FACE_NAMES[hidden_face_idx]}")
    
    # Get components
    patterns, center, corners, edges = solve_hidden_face(
        visible_faces, hidden_face_idx, return_components=True
    )
    
    print(f"\nComponents:")
    print(f"  Center: {COLOR_NAMES[center]}")
    print(f"  Corner solutions: {len(corners)}")
    print(f"  Edge solutions: {len(edges)}")
    print(f"  Combined patterns: {len(patterns)}")
    
    print(f"\nTarget pattern:")
    print(pattern_to_string(target))
    
    found, idx = check_solution(patterns, target)
    print(f"\nTarget found: {found}" + (f" (index {idx})" if found else ""))
    
    # Show first few solutions
    print(f"\nFirst 3 solution patterns:")
    for i, p in enumerate(patterns[:3]):
        print(f"\n  Solution {i}:")
        for line in pattern_to_string(p).split('\n'):
            print(f"    {line}")
    
    if len(patterns) > 3:
        print(f"\n  ... and {len(patterns) - 3} more")


def test_single(sample_idx: int = 0):
    """Test a single sample with full debug output."""
    print("=" * 60)
    print(f"TESTING SINGLE SAMPLE: {sample_idx}")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent / "data"
    data = np.load(data_dir / "test.npz")
    
    debug_sample(
        data['visible'][sample_idx],
        data['hidden_idx'][sample_idx],
        data['target'][sample_idx]
    )


if __name__ == "__main__":
    test()