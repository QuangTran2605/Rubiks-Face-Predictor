"""
solver_edges.py - Edge solver with piece-to-edge matching

Strategy:
1. Find 8 fully visible edges -> identify 8 used pieces
2. Remaining 4 pieces are on hidden face
3. For each partially visible edge (1 sticker visible), find which remaining pieces could match
4. Try all valid piece-to-edge assignments
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

# Edge sticker positions (face, flat_index)
# Each edge has 2 stickers on adjacent faces
EDGES = [
    [(U, 1), (B, 1)],   # 0: UB
    [(U, 3), (L, 1)],   # 1: UL
    [(U, 5), (R, 1)],   # 2: UR
    [(U, 7), (F, 1)],   # 3: UF
    [(D, 1), (F, 7)],   # 4: DF
    [(D, 3), (L, 7)],   # 5: DL
    [(D, 5), (R, 7)],   # 6: DR
    [(D, 7), (B, 7)],   # 7: DB
    [(F, 3), (L, 5)],   # 8: FL
    [(F, 5), (R, 3)],   # 9: FR
    [(B, 3), (R, 5)],   # 10: BR
    [(B, 5), (L, 3)],   # 11: BL
]

# The 12 valid edge color combinations
# Opposite colors (W-Y, G-B, O-R) cannot share an edge
VALID_EDGE_SETS = [
    frozenset([0, 2]),  # 0: W, G
    frozenset([0, 3]),  # 1: W, B
    frozenset([0, 4]),  # 2: W, O
    frozenset([0, 5]),  # 3: W, R
    frozenset([1, 2]),  # 4: Y, G
    frozenset([1, 3]),  # 5: Y, B
    frozenset([1, 4]),  # 6: Y, O
    frozenset([1, 5]),  # 7: Y, R
    frozenset([2, 4]),  # 8: G, O
    frozenset([2, 5]),  # 9: G, R
    frozenset([3, 4]),  # 10: B, O
    frozenset([3, 5]),  # 11: B, R
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


def get_edge_sticker_colors(lookup: Dict, edge_idx: int, hidden_face_idx: int) -> Tuple[List[int], int]:
    """
    Get visible colors and hidden position for an edge.
    
    Returns:
        (visible_colors, hidden_pos)
        - visible_colors: list of visible sticker colors (1 or 2 items)
        - hidden_pos: position on hidden face (-1 if fully visible)
    """
    stickers = EDGES[edge_idx]
    
    visible_colors = []
    hidden_pos = -1
    
    for face, pos in stickers:
        if face == hidden_face_idx:
            hidden_pos = pos
        else:
            visible_colors.append(lookup[(face, pos)])
    
    return visible_colors, hidden_pos


def identify_edge_piece(colors: Set[int]) -> Optional[int]:
    """Identify which edge piece has these colors. Returns piece index or None."""
    color_set = frozenset(colors)
    for i, valid_set in enumerate(VALID_EDGE_SETS):
        if color_set == valid_set:
            return i
    return None


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_edges(visible_faces: np.ndarray, hidden_face_idx: int) -> List[Dict[int, int]]:
    """
    Solve edge stickers on hidden face.
    
    Returns:
        List of valid solutions, each is {position: color} for positions 1, 3, 5, 7
    """
    lookup = build_sticker_lookup(visible_faces, hidden_face_idx)
    
    # Step 1: Identify fully visible edges and their pieces
    fully_visible_edges = []      # [(edge_idx, piece_idx), ...]
    partially_visible_edges = []  # [(edge_idx, visible_color, hidden_pos), ...]
    
    used_pieces = set()
    
    for edge_idx in range(12):
        visible_colors, hidden_pos = get_edge_sticker_colors(lookup, edge_idx, hidden_face_idx)
        
        if hidden_pos == -1:
            # Fully visible - both colors visible
            piece_idx = identify_edge_piece(set(visible_colors))
            if piece_idx is None:
                return []  # Invalid cube
            fully_visible_edges.append((edge_idx, piece_idx))
            used_pieces.add(piece_idx)
        else:
            # Partially visible - 1 color visible, 1 hidden
            partially_visible_edges.append((edge_idx, visible_colors[0], hidden_pos))
    
    if len(fully_visible_edges) != 8 or len(partially_visible_edges) != 4:
        return []
    
    # Step 2: Remaining pieces are on hidden face
    remaining_pieces = set(range(12)) - used_pieces
    
    if len(remaining_pieces) != 4:
        return []
    
    # Step 3: For each partially visible edge, find which pieces could match
    # Build: edge_idx -> [(piece_idx, hidden_color), ...]
    edge_to_possible_pieces = {}
    
    for edge_idx, visible_color, hidden_pos in partially_visible_edges:
        possible = []
        
        for piece_idx in remaining_pieces:
            piece_colors = VALID_EDGE_SETS[piece_idx]
            
            if visible_color in piece_colors:
                hidden_color = next(iter(piece_colors - {visible_color}))
                possible.append((piece_idx, hidden_color))
        
        if not possible:
            return []  # No valid piece for this edge
        
        edge_to_possible_pieces[edge_idx] = possible
    
    # Step 4: Try all valid assignments of pieces to edges
    # Each piece can only be used once
    
    partial_edges = [e[0] for e in partially_visible_edges]
    edge_hidden_pos = {e[0]: e[2] for e in partially_visible_edges}
    
    valid_solutions = []
    
    def try_assignment(edge_list_idx: int, used: Set[int], current_assignment: Dict[int, int]):
        """Recursively try piece assignments."""
        if edge_list_idx == len(partial_edges):
            # Complete assignment found
            valid_solutions.append(current_assignment.copy())
            return
        
        edge_idx = partial_edges[edge_list_idx]
        hidden_pos = edge_hidden_pos[edge_idx]
        
        for piece_idx, hidden_color in edge_to_possible_pieces[edge_idx]:
            if piece_idx in used:
                continue  # Piece already used
            
            # Try this assignment
            used.add(piece_idx)
            current_assignment[hidden_pos] = hidden_color
            
            try_assignment(edge_list_idx + 1, used, current_assignment)
            
            # Backtrack
            used.remove(piece_idx)
            del current_assignment[hidden_pos]
    
    try_assignment(0, set(), {})
    
    return valid_solutions


# =============================================================================
# TEST
# =============================================================================

def test():
    print("=" * 60)
    print("TESTING EDGE SOLVER")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / "train.npz")
    
    visible = data['visible']
    target = data['target']
    hidden_idx = data['hidden_idx']
    
    n = len(target)
    edge_positions = [1, 3, 5, 7]
    
    # Statistics
    total_samples = 0
    samples_with_solution = 0
    samples_with_unique = 0
    samples_with_multiple = 0
    samples_correct = 0
    solution_count_dist = {}
    
    errors = []
    
    for i in range(n):
        total_samples += 1
        
        solutions = solve_edges(visible[i], hidden_idx[i])
        
        # Track solution count distribution
        count = len(solutions)
        solution_count_dist[count] = solution_count_dist.get(count, 0) + 1
        
        if count == 0:
            errors.append((i, "No solution"))
            continue
        
        samples_with_solution += 1
        
        if count == 1:
            samples_with_unique += 1
        else:
            samples_with_multiple += 1
        
        # Check if target is in solutions
        target_edges = {pos: int(target[i][pos]) for pos in edge_positions}
        
        found = False
        for sol in solutions:
            if sol == target_edges:
                found = True
                break
        
        if found:
            samples_correct += 1
        else:
            errors.append((i, f"Target not in {count} solutions"))
    
    print(f"\nResults:")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples with solution: {samples_with_solution}")
    print(f"  Samples with unique solution: {samples_with_unique}")
    print(f"  Samples with multiple solutions: {samples_with_multiple}")
    print(f"  Samples where target in solutions: {samples_correct}")
    print(f"  Accuracy: {samples_correct / total_samples:.4f}")
    
    print(f"\nSolution count distribution:")
    for count in sorted(solution_count_dist.keys()):
        freq = solution_count_dist[count]
        pct = 100 * freq / total_samples
        print(f"  {count} solutions: {freq} samples ({pct:.1f}%)")
    
    if samples_correct == total_samples:
        print("\n✓ EDGE SOLVER: PERFECT!")
    else:
        print(f"\n✗ EDGE SOLVER: {total_samples - samples_correct} errors")
        
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
    
    edge_positions = [1, 3, 5, 7]
    target_edges = {pos: int(target[pos]) for pos in edge_positions}
    print(f"Target edges: {target_edges}")
    print(f"Target edge colors: {[COLOR_NAMES[target_edges[p]] for p in edge_positions]}")
    
    lookup = build_sticker_lookup(visible_faces, hidden_face_idx)
    
    print(f"\nEdge analysis:")
    used_pieces = set()
    
    for edge_idx in range(12):
        visible_colors, hidden_pos = get_edge_sticker_colors(lookup, edge_idx, hidden_face_idx)
        vis_names = [COLOR_NAMES[c] for c in visible_colors]
        
        if hidden_pos == -1:
            piece_idx = identify_edge_piece(set(visible_colors))
            print(f"  Edge {edge_idx}: FULLY VISIBLE, colors={vis_names}, piece={piece_idx}")
            if piece_idx is not None:
                used_pieces.add(piece_idx)
        else:
            actual_hidden = target[hidden_pos]
            all_colors = set(visible_colors) | {actual_hidden}
            actual_piece = identify_edge_piece(all_colors)
            
            # Find matching pieces
            remaining = set(range(12)) - used_pieces
            matching = []
            for p in remaining:
                if visible_colors[0] in VALID_EDGE_SETS[p]:
                    hc = next(iter(VALID_EDGE_SETS[p] - set(visible_colors)))
                    matching.append((p, COLOR_NAMES[hc]))
            
            print(f"  Edge {edge_idx}: visible={vis_names}, hidden_pos={hidden_pos}, actual={COLOR_NAMES[actual_hidden]}, actual_piece={actual_piece}")
            print(f"    Matching pieces from remaining: {matching}")
    
    print(f"\nUsed pieces: {used_pieces}")
    print(f"Remaining pieces: {set(range(12)) - used_pieces}")


if __name__ == "__main__":
    test()