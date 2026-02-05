"""
cube.py - Rubik's Cube Simulator

A complete 3x3 Rubik's Cube implementation for ML purposes.
Supports all standard moves and scrambling.

"""

import numpy as np
from typing import List, Tuple, Optional
import random


class RubiksCube:
    """
    A 3x3 Rubik's Cube simulator.
    
    Face indices:
        0: Up (U)    - White  (in solved state)
        1: Down (D)  - Yellow
        2: Front (F) - Green
        3: Back (B)  - Blue
        4: Left (L)  - Orange
        5: Right (R) - Red
    
    State representation:
        - Shape: (6, 3, 3) numpy array
        - Values: 0-5 representing colors
        - state[face_idx, row, col] gives the color at that position
    """
    #Class constants for faces: 
    U, D, F, B, L, R = 0, 1, 2, 3, 4, 5
    FACE_NAMES = ['U', 'D', 'F', 'B', 'L', 'R']

    # Move definitions: (face_to_rotate, [4 adjacent edges that cycle])
    # Each edge is (face, row_slice, col_slice, needs_reverse)
    # Edges cycle: 0 -> 1 -> 2 -> 3 -> 0 (for clockwise)
    MOVES = {
        'U': (0, [
            (2, 0, slice(None), False),  # F top row
            (5, 0, slice(None), False),  # R top row
            (3, 0, slice(None), False),  # B top row
            (4, 0, slice(None), False)   # L top row
        ]),
        'D': (1, [
            (2, 2, slice(None), False),  # F bottom row
            (4, 2, slice(None), False),  # L bottom row
            (3, 2, slice(None), False),  # B bottom row
            (5, 2, slice(None), False)   # R bottom row
        ]),
        'F': (2, [
            (0, 2, slice(None), False),  # U bottom row
            (5, slice(None), 0, True),   # R left col
            (1, 0, slice(None), False),  # D top row
            (4, slice(None), 2, True)    # L right col
        ]),
        'B': (3, [
            (0, 0, slice(None), False),  # U top row
            (4, slice(None), 0, True),   # L left col
            (1, 2, slice(None), False),  # D bottom row
            (5, slice(None), 2, True)    # R right col
        ]),
        'L': (4, [
            (0, slice(None), 0, True),   # U left col
            (2, slice(None), 0, True),   # F left col
            (1, slice(None), 0, True),   # D left col
            (3, slice(None), 2, True)    # B right col
        ]),
        'R': (5, [
            (0, slice(None), 2, True),   # U right col
            (3, slice(None), 0, True),   # B left col
            (1, slice(None), 2, True),   # D right col
            (2, slice(None), 2, True)    # F right col
        ])
    }

    def __init__(self, state: Optional[np.ndarray] = None):
        """
        Initialize the Rubik's Cube.
        
        Args:
            state (Optional[np.ndarray]): Initial state of the cube. If None, initializes to solved state.
        """
        if state is not None: 
            self.state = state.copy()
        else:
            self.reset()
    
    def reset(self) -> None:
        """Reset cube to solved state."""
        # Each face filled with its index (0-5)
        self.state = np.zeros((6, 3, 3), dtype=np.int8)
        for face_idx in range(6):
            self.state[face_idx] = face_idx
    
    def copy(self) -> 'RubiksCube':
        """Create a deep copy of this cube."""
        return RubiksCube(state=self.state.copy())
    
    def get_face(self, face: int) -> np.ndarray:
        """Get a face as 3x3 array."""
        return self.state[face].copy()
    
    def get_all_faces(self) -> np.ndarray:
        """Get all faces as (6, 9) array."""
        return self.state.reshape(6, 9)
    
    def _get_edge(self, face: int, row, col) -> np.ndarray:
        """Extract an edge (row or column) from a face."""
        return self.state[face, row, col].copy()
    
    def _set_edge(self, face: int, row, col, values: np.ndarray) -> None:
        """Set an edge (row or column) on a face."""
        self.state[face, row, col] = values
    
    def _do_move(self, move_name: str, prime: bool = False) -> None:
        """
        Execute a single face move.
        
        Args:
            move_name: One of 'U', 'D', 'F', 'B', 'L', 'R'
            prime: If True, rotate counter-clockwise
        """
        face, edges = self.MOVES[move_name]
        
        # Rotate the face itself
        k = 1 if prime else -1  # rot90 direction
        self.state[face] = np.rot90(self.state[face], k=k)
        
        # Collect current edge values
        edge_values = []
        for f, r, c, rev in edges:
            val = self._get_edge(f, r, c)
            edge_values.append(val)
        
        # Cycle edges (direction depends on prime)
        n = len(edges)
        if prime:
            # Cycle: 0 <- 1 <- 2 <- 3 <- 0
            new_order = [1, 2, 3, 0]
        else:
            # Cycle: 0 -> 1 -> 2 -> 3 -> 0  means 0 <- 3, 1 <- 0, etc.
            new_order = [3, 0, 1, 2]
        
        # Apply cycled values with proper reversals
        for i, (f, r, c, rev) in enumerate(edges):
            src_idx = new_order[i]
            val = edge_values[src_idx]
            
            # Handle reversal based on source and destination
            src_rev = edges[src_idx][3]
            if src_rev != rev:
                val = val[::-1]
            
            self._set_edge(f, r, c, val)
    
    def execute_move(self, move: str) -> None:
        """
        Execute a move in standard notation.
        
        Examples: 'U', "U'", 'U2', 'R', "R'", 'R2'
        """
        if len(move) == 1:
            self._do_move(move, prime=False)
        elif move[1] == "'":
            self._do_move(move[0], prime=True)
        elif move[1] == "2":
            self._do_move(move[0], prime=False)
            self._do_move(move[0], prime=False)
        else:
            raise ValueError(f"Unknown move: {move}")
    
    def execute_moves(self, moves: List[str]) -> None:
        """Execute a sequence of moves."""
        for move in moves:
            self.execute_move(move)
    
    def scramble(self, num_moves: int = 20, seed: Optional[int] = None) -> List[str]:
        """
        Scramble with random moves.
        
        Returns list of moves applied.
        """
        if seed is not None:
            random.seed(seed)
        
        all_moves = ['U', "U'", 'D', "D'", 'F', "F'", 
                     'B', "B'", 'L', "L'", 'R', "R'"]
        
        moves_applied = []
        last_face = None
        
        for _ in range(num_moves):
            # Avoid same face twice in a row
            available = [m for m in all_moves if m[0] != last_face]
            move = random.choice(available)
            moves_applied.append(move)
            last_face = move[0]
        
        self.execute_moves(moves_applied)
        return moves_applied
    
    def is_valid(self) -> bool:
        """Check if cube has correct color counts (9 of each)."""
        flat = self.state.flatten()
        return all(np.sum(flat == c) == 9 for c in range(6))
    
    def __repr__(self) -> str:
        """Simple text display of the cube."""
        colors = ['W', 'Y', 'G', 'B', 'O', 'R']  # Color chars
        
        def face_str(f, row):
            return ' '.join(colors[c] for c in self.state[f, row])
        
        lines = []
        
        # Up face (indented)
        for row in range(3):
            lines.append(f"       {face_str(0, row)}")
        
        lines.append("")
        
        # L F R B in a row
        for row in range(3):
            l = face_str(4, row)
            f = face_str(2, row)
            r = face_str(5, row)
            b = face_str(3, row)
            lines.append(f"{l}  {f}  {r}  {b}")
        
        lines.append("")
        
        # Down face (indented)
        for row in range(3):
            lines.append(f"       {face_str(1, row)}")
        
        return '\n'.join(lines)
    
def create_scrambled_cube(num_moves: int, seed: Optional[int] = None) -> Tuple[RubiksCube, List[str]]:
    """Create a new scrambled cube. Returns (cube, moves_applied)."""
    cube = RubiksCube()
    moves = cube.scramble(num_moves, seed)
    return cube, moves

if __name__ == "__main__":
    # Quick demo
    print("Solved cube:")
    cube = RubiksCube()
    print(cube)
    print()
        
    print("After scrambling:")
    moves = cube.scramble(20, seed=42)
    print(f"Moves: {' '.join(moves)}")
    print(cube)
    print(f"\nValid: {cube.is_valid()}")
    
    


    