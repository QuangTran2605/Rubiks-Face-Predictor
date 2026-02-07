"""
solver_corners.py - Corner solver with proper piece-to-corner matching

Strategy:
1. Find 4 fully visible corners -> identify 4 used pieces
2. Remaining 4 pieces are on hidden face
3. For each partially visible corner, find which remaining pieces could match
4. Try all valid piece-to-corner assignments
5. For each assignment, the hidden color is determined
6. Return all valid solutions
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from itertools import permutations

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

# Corner sticker positions
CORNERS = [
    [(U, 0), (L, 0), (B, 2)],   # 0: ULB
    [(U, 2), (B, 0), (R, 2)],   # 1: UBR
    [(U, 6), (F, 0), (L, 2)],   # 2: UFL
    [(U, 8), (R, 0), (F, 2)],   # 3: UFR
    [(D, 0), (L, 8), (F, 6)],   # 4: DLF
    [(D, 2), (F, 8), (R, 6)],   # 5: DFR
    [(D, 6), (B, 8), (L, 6)],   # 6: DBL
    [(D, 8), (R, 8), (B, 6)],   # 7: DBR
]

# The 8 valid corner color combinations
VALID_CORNER_SETS = [
    frozenset([0, 3, 4]),  # 0: W, B, O
    frozenset([0, 3, 5]),  # 1: W, B, R
    frozenset([0, 2, 4]),  # 2: W, G, O
    frozenset([0, 2, 5]),  # 3: W, G, R
    frozenset([1, 2, 4]),  # 4: Y, G, O
    frozenset([1, 2, 5]),  # 5: Y, G, R
    frozenset([1, 3, 4]),  # 6: Y, B, O
    frozenset([1, 3, 5]),  # 7: Y, B, R
]

# =============================================================================
# HELPERS
# =============================================================================

def build_sticker_lookup(visible_faces: np.ndarray, hidden_face_idx: int) -> Dict[Tuple[int, int], int]:
    """Build lookup: (face, position) -> color for all visible stickers."""
    lookup = {}
    visible_face_list = [i for i in range(6) if i != hidden_face_idx]
    
    for i, face_idx in enumerate(visible_face_list):
        for pos in range(9):
            lookup[(face_idx, pos)] = int(visible_faces[i][pos])
    
    return lookup


def get_corner_sticker_colors(lookup: Dict, corner_idx: int, hidden_face_idx: int) -> Tuple[List[int], int]:
    """
    Get visible colors and hidden position for a corner.
    
    Returns:
        (visible_colors, hidden_pos)
        - visible_colors: list of visible sticker colors (2 or 3 items)
        - hidden_pos: position on hidden face (-1 if fully visible)
    """
    stickers = CORNERS[corner_idx]
    
    visible_colors = []
    hidden_pos = -1
    
    for face, pos in stickers:
        if face == hidden_face_idx:
            hidden_pos = pos
        else:
            visible_colors.append(lookup[(face, pos)])
    
    return visible_colors, hidden_pos


def identify_piece(colors: Set[int]) -> Optional[int]:
    """Identify which piece has these colors. Returns piece index or None."""
    color_set = frozenset(colors)
    for i, valid_set in enumerate(VALID_CORNER_SETS):
        if color_set == valid_set:
            return i
    return None


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_corners(visible_faces: np.ndarray, hidden_face_idx: int) -> List[Dict[int, int]]:
    """
    Solve corner stickers on hidden face.
    
    Returns:
        List of valid solutions, each is {position: color} for positions 0, 2, 6, 8
    """
    lookup = build_sticker_lookup(visible_faces, hidden_face_idx)
    
    # Step 1: Identify fully visible corners and their pieces
    fully_visible_corners = []  # [(corner_idx, piece_idx), ...]
    partially_visible_corners = []  # [(corner_idx, visible_colors, hidden_pos), ...]
    
    used_pieces = set()
    
    for corner_idx in range(8):
        visible_colors, hidden_pos = get_corner_sticker_colors(lookup, corner_idx, hidden_face_idx)
        
        if hidden_pos == -1:
            # Fully visible - all 3 colors visible
            piece_idx = identify_piece(set(visible_colors))
            if piece_idx is None:
                return []  # Invalid cube
            fully_visible_corners.append((corner_idx, piece_idx))
            used_pieces.add(piece_idx)
        else:
            # Partially visible - 2 colors visible, 1 hidden
            partially_visible_corners.append((corner_idx, visible_colors, hidden_pos))
    
    if len(fully_visible_corners) != 4 or len(partially_visible_corners) != 4:
        return []
    
    # Step 2: Remaining pieces are on hidden face
    remaining_pieces = set(range(8)) - used_pieces
    
    if len(remaining_pieces) != 4:
        return []
    
    # Step 3: For each partially visible corner, find which pieces could match
    # Build: corner_idx -> [(piece_idx, hidden_color), ...]
    corner_to_possible_pieces = {}
    
    for corner_idx, visible_colors, hidden_pos in partially_visible_corners:
        visible_set = set(visible_colors)
        possible = []
        
        for piece_idx in remaining_pieces:
            piece_colors = VALID_CORNER_SETS[piece_idx]
            
            if visible_set.issubset(piece_colors):
                hidden_color = next(iter(piece_colors - visible_set))
                possible.append((piece_idx, hidden_color))
        
        if not possible:
            return []  # No valid piece for this corner
        
        corner_to_possible_pieces[corner_idx] = possible
    
    # Step 4: Try all valid assignments of pieces to corners
    # Each piece can only be used once
    
    partial_corners = [c[0] for c in partially_visible_corners]
    corner_hidden_pos = {c[0]: c[2] for c in partially_visible_corners}
    
    valid_solutions = []
    
    def try_assignment(corner_list_idx: int, used: Set[int], current_assignment: Dict[int, int]):
        """Recursively try piece assignments."""
        if corner_list_idx == len(partial_corners):
            # Complete assignment found
            valid_solutions.append(current_assignment.copy())
            return
        
        corner_idx = partial_corners[corner_list_idx]
        hidden_pos = corner_hidden_pos[corner_idx]
        
        for piece_idx, hidden_color in corner_to_possible_pieces[corner_idx]:
            if piece_idx in used:
                continue  # Piece already used
            
            # Try this assignment
            used.add(piece_idx)
            current_assignment[hidden_pos] = hidden_color
            
            try_assignment(corner_list_idx + 1, used, current_assignment)
            
            # Backtrack
            used.remove(piece_idx)
            del current_assignment[hidden_pos]
    
    try_assignment(0, set(), {})
    
    return valid_solutions


# =============================================================================
# TEST
# =============================================================================

def test():
    print("="*60)
    print("TESTING CORNER SOLVER")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / "train.npz")
    
    visible = data['visible']
    target = data['target']
    hidden_idx = data['hidden_idx']
    
    n = len(target)
    corner_positions = [0, 2, 6, 8]
    
    # Statistics
    total_samples = 0
    samples_with_solution = 0
    samples_with_unique = 0
    samples_with_multiple = 0
    samples_correct = 0
    
    errors = []
    
    for i in range(n):
        total_samples += 1
        
        solutions = solve_corners(visible[i], hidden_idx[i])
        
        if len(solutions) == 0:
            errors.append((i, "No solution"))
            continue
        
        samples_with_solution += 1
        
        if len(solutions) == 1:
            samples_with_unique += 1
        else:
            samples_with_multiple += 1
        
        # Check if target is in solutions
        target_corners = {pos: int(target[i][pos]) for pos in corner_positions}
        
        found = False
        for sol in solutions:
            if sol == target_corners:
                found = True
                break
        
        if found:
            samples_correct += 1
        else:
            errors.append((i, f"Target not in {len(solutions)} solutions"))
    
    print(f"\nResults:")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples with solution: {samples_with_solution}")
    print(f"  Samples with unique solution: {samples_with_unique}")
    print(f"  Samples with multiple solutions: {samples_with_multiple}")
    print(f"  Samples where target in solutions: {samples_correct}")
    print(f"  Accuracy: {samples_correct/total_samples:.4f}")
    
    if samples_correct == total_samples:
        print("\n✓ CORNER SOLVER: PERFECT!")
    else:
        print(f"\n✗ CORNER SOLVER: {total_samples - samples_correct} errors")
        
        # Show first few errors
        print(f"\nFirst 5 errors:")
        for i, reason in errors[:5]:
            print(f"  Sample {i}: {reason}")
        
        # Debug first error
        if errors:
            first_error_idx = errors[0][0]
            debug_sample(visible[first_error_idx], hidden_idx[first_error_idx], target[first_error_idx])


def debug_sample(visible_faces: np.ndarray, hidden_face_idx: int, target: np.ndarray):
    """Debug a single sample."""
    print(f"\n--- Debug Sample ---")
    print(f"Hidden face: {hidden_face_idx} ({FACE_NAMES[hidden_face_idx]})")
    
    corner_positions = [0, 2, 6, 8]
    target_corners = {pos: int(target[pos]) for pos in corner_positions}
    print(f"Target corners: {target_corners}")
    print(f"Target corner colors: {[COLOR_NAMES[target_corners[p]] for p in corner_positions]}")
    
    lookup = build_sticker_lookup(visible_faces, hidden_face_idx)
    
    print(f"\nCorner analysis:")
    used_pieces = set()
    
    for corner_idx in range(8):
        visible_colors, hidden_pos = get_corner_sticker_colors(lookup, corner_idx, hidden_face_idx)
        vis_names = [COLOR_NAMES[c] for c in visible_colors]
        
        if hidden_pos == -1:
            piece_idx = identify_piece(set(visible_colors))
            print(f"  Corner {corner_idx}: FULLY VISIBLE, colors={vis_names}, piece={piece_idx}")
            if piece_idx is not None:
                used_pieces.add(piece_idx)
        else:
            actual_hidden = target[hidden_pos]
            all_colors = set(visible_colors) | {actual_hidden}
            actual_piece = identify_piece(all_colors)
            
            # Find matching pieces
            remaining = set(range(8)) - used_pieces
            matching = []
            for p in remaining:
                if set(visible_colors).issubset(VALID_CORNER_SETS[p]):
                    hc = next(iter(VALID_CORNER_SETS[p] - set(visible_colors)))
                    matching.append((p, COLOR_NAMES[hc]))
            
            print(f"  Corner {corner_idx}: visible={vis_names}, hidden_pos={hidden_pos}, actual={COLOR_NAMES[actual_hidden]}, actual_piece={actual_piece}")
            print(f"    Matching pieces from remaining: {matching}")
    
    print(f"\nUsed pieces: {used_pieces}")
    print(f"Remaining pieces: {set(range(8)) - used_pieces}")


if __name__ == "__main__":
    test()