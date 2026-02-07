"""
rubiks_cube.py - Rubik's Cube Simulator

Uses explicit, verified move implementations.
Each move is carefully defined with correct sticker cycles.
"""

import numpy as np
from typing import List, Tuple, Optional
import random


class RubiksCube:
    """
    A 3x3 Rubik's Cube simulator.
    
    Faces: U=0, D=1, F=2, B=3, L=4, R=5
    
    State: (6, 3, 3) numpy array
    
    Each face layout:
        [0][1][2]
        [3][4][5]
        [6][7][8]
    
    When looking at the cube:
        - U (Up/White) is on top
        - F (Front/Green) faces you
        - R (Right/Red) is on your right
        - L (Left/Orange) is on your left
        - B (Back/Blue) is behind
        - D (Down/Yellow) is on bottom
    """
    
    U, D, F, B, L, R = 0, 1, 2, 3, 4, 5
    FACE_NAMES = ['U', 'D', 'F', 'B', 'L', 'R']
    COLOR_NAMES = ['W', 'Y', 'G', 'B', 'O', 'R']
    
    def __init__(self, state: Optional[np.ndarray] = None):
        if state is not None:
            self.state = state.copy()
        else:
            self.reset()
    
    def reset(self) -> None:
        """Reset to solved state."""
        self.state = np.zeros((6, 3, 3), dtype=np.int8)
        for i in range(6):
            self.state[i] = i
    
    def copy(self) -> 'RubiksCube':
        return RubiksCube(self.state)
    
    def get_face(self, face: int) -> np.ndarray:
        return self.state[face].copy()
    
    def get_all_faces(self) -> np.ndarray:
        """Get all faces as (6, 9) array."""
        return self.state.reshape(6, 9)
    
    def is_valid(self) -> bool:
        """Check color counts (9 of each)."""
        flat = self.state.flatten()
        return all(np.sum(flat == c) == 9 for c in range(6))
    
    # =========================================================================
    # ROTATION HELPERS
    # =========================================================================
    
    def _rotate_cw(self, face: int) -> None:
        """Rotate face 90° clockwise."""
        self.state[face] = np.rot90(self.state[face], k=-1)
    
    def _rotate_ccw(self, face: int) -> None:
        """Rotate face 90° counter-clockwise."""
        self.state[face] = np.rot90(self.state[face], k=1)
    
    # =========================================================================
    # MOVE IMPLEMENTATIONS
    # Each move rotates one face and cycles 4 adjacent strips
    # =========================================================================
    
    def move_U(self) -> None:
        """U move: rotate Up face clockwise."""
        self._rotate_cw(self.U)
        
        # Cycle: F[top] -> L[top] -> B[top] -> R[top] -> F[top]
        temp = self.state[self.F, 0, :].copy()
        self.state[self.F, 0, :] = self.state[self.R, 0, :]
        self.state[self.R, 0, :] = self.state[self.B, 0, :]
        self.state[self.B, 0, :] = self.state[self.L, 0, :]
        self.state[self.L, 0, :] = temp
    
    def move_U_prime(self) -> None:
        """U' move: rotate Up face counter-clockwise."""
        self._rotate_ccw(self.U)
        
        # Cycle: F[top] -> R[top] -> B[top] -> L[top] -> F[top]
        temp = self.state[self.F, 0, :].copy()
        self.state[self.F, 0, :] = self.state[self.L, 0, :]
        self.state[self.L, 0, :] = self.state[self.B, 0, :]
        self.state[self.B, 0, :] = self.state[self.R, 0, :]
        self.state[self.R, 0, :] = temp
    
    def move_D(self) -> None:
        """D move: rotate Down face clockwise."""
        self._rotate_cw(self.D)
        
        # Cycle: F[bottom] -> R[bottom] -> B[bottom] -> L[bottom] -> F[bottom]
        # Note: opposite direction from U because we're looking from below
        temp = self.state[self.F, 2, :].copy()
        self.state[self.F, 2, :] = self.state[self.L, 2, :]
        self.state[self.L, 2, :] = self.state[self.B, 2, :]
        self.state[self.B, 2, :] = self.state[self.R, 2, :]
        self.state[self.R, 2, :] = temp
    
    def move_D_prime(self) -> None:
        """D' move: rotate Down face counter-clockwise."""
        self._rotate_ccw(self.D)
        
        temp = self.state[self.F, 2, :].copy()
        self.state[self.F, 2, :] = self.state[self.R, 2, :]
        self.state[self.R, 2, :] = self.state[self.B, 2, :]
        self.state[self.B, 2, :] = self.state[self.L, 2, :]
        self.state[self.L, 2, :] = temp
    
    def move_F(self) -> None:
        """F move: rotate Front face clockwise."""
        self._rotate_cw(self.F)
        
        # Cycle: U[bottom] -> R[left] -> D[top] -> L[right] -> U[bottom]
        temp = self.state[self.U, 2, :].copy()
        
        self.state[self.U, 2, :] = self.state[self.L, :, 2][::-1]
        self.state[self.L, :, 2] = self.state[self.D, 0, :]
        self.state[self.D, 0, :] = self.state[self.R, :, 0][::-1]
        self.state[self.R, :, 0] = temp
    
    def move_F_prime(self) -> None:
        """F' move: rotate Front face counter-clockwise."""
        self._rotate_ccw(self.F)
        
        temp = self.state[self.U, 2, :].copy()
        
        self.state[self.U, 2, :] = self.state[self.R, :, 0]
        self.state[self.R, :, 0] = self.state[self.D, 0, :][::-1]
        self.state[self.D, 0, :] = self.state[self.L, :, 2]
        self.state[self.L, :, 2] = temp[::-1]
    
    def move_B(self) -> None:
        """B move: rotate Back face clockwise."""
        self._rotate_cw(self.B)
        
        # Cycle: U[top] -> L[left] -> D[bottom] -> R[right] -> U[top]
        temp = self.state[self.U, 0, :].copy()
        
        self.state[self.U, 0, :] = self.state[self.R, :, 2]
        self.state[self.R, :, 2] = self.state[self.D, 2, :][::-1]
        self.state[self.D, 2, :] = self.state[self.L, :, 0]
        self.state[self.L, :, 0] = temp[::-1]
    
    def move_B_prime(self) -> None:
        """B' move: rotate Back face counter-clockwise."""
        self._rotate_ccw(self.B)
        
        temp = self.state[self.U, 0, :].copy()
        
        self.state[self.U, 0, :] = self.state[self.L, :, 0][::-1]
        self.state[self.L, :, 0] = self.state[self.D, 2, :]
        self.state[self.D, 2, :] = self.state[self.R, :, 2][::-1]
        self.state[self.R, :, 2] = temp
    
    def move_L(self) -> None:
        """L move: rotate Left face clockwise."""
        self._rotate_cw(self.L)
        
        # Cycle: U[left] -> F[left] -> D[left] -> B[right] -> U[left]
        temp = self.state[self.U, :, 0].copy()
        
        self.state[self.U, :, 0] = self.state[self.B, :, 2][::-1]
        self.state[self.B, :, 2] = self.state[self.D, :, 0][::-1]
        self.state[self.D, :, 0] = self.state[self.F, :, 0]
        self.state[self.F, :, 0] = temp
    
    def move_L_prime(self) -> None:
        """L' move: rotate Left face counter-clockwise."""
        self._rotate_ccw(self.L)
        
        temp = self.state[self.U, :, 0].copy()
        
        self.state[self.U, :, 0] = self.state[self.F, :, 0]
        self.state[self.F, :, 0] = self.state[self.D, :, 0]
        self.state[self.D, :, 0] = self.state[self.B, :, 2][::-1]
        self.state[self.B, :, 2] = temp[::-1]
    
    def move_R(self) -> None:
        """R move: rotate Right face clockwise."""
        self._rotate_cw(self.R)
        
        # Cycle: U[right] -> B[left] -> D[right] -> F[right] -> U[right]
        temp = self.state[self.U, :, 2].copy()
        
        self.state[self.U, :, 2] = self.state[self.F, :, 2]
        self.state[self.F, :, 2] = self.state[self.D, :, 2]
        self.state[self.D, :, 2] = self.state[self.B, :, 0][::-1]
        self.state[self.B, :, 0] = temp[::-1]
    
    def move_R_prime(self) -> None:
        """R' move: rotate Right face counter-clockwise."""
        self._rotate_ccw(self.R)
        
        temp = self.state[self.U, :, 2].copy()
        
        self.state[self.U, :, 2] = self.state[self.B, :, 0][::-1]
        self.state[self.B, :, 0] = self.state[self.D, :, 2][::-1]
        self.state[self.D, :, 2] = self.state[self.F, :, 2]
        self.state[self.F, :, 2] = temp
    
    # =========================================================================
    # MOVE EXECUTION
    # =========================================================================
    
    def execute_move(self, move: str) -> None:
        """Execute a move in standard notation."""
        move_map = {
            'U': self.move_U,
            "U'": self.move_U_prime,
            'D': self.move_D,
            "D'": self.move_D_prime,
            'F': self.move_F,
            "F'": self.move_F_prime,
            'B': self.move_B,
            "B'": self.move_B_prime,
            'L': self.move_L,
            "L'": self.move_L_prime,
            'R': self.move_R,
            "R'": self.move_R_prime,
        }
        
        if move in move_map:
            move_map[move]()
        elif len(move) == 2 and move[1] == '2':
            # Double move (e.g., U2)
            base = move[0]
            move_map[base]()
            move_map[base]()
        else:
            raise ValueError(f"Unknown move: {move}")
    
    def execute_moves(self, moves: List[str]) -> None:
        """Execute a sequence of moves."""
        for move in moves:
            self.execute_move(move)
    
    def scramble(self, num_moves: int = 20, seed: Optional[int] = None) -> List[str]:
        """Scramble with random moves."""
        if seed is not None:
            random.seed(seed)
        
        all_moves = ['U', "U'", 'D', "D'", 'F', "F'", 
                     'B', "B'", 'L', "L'", 'R', "R'"]
        
        moves_applied = []
        last_face = None
        
        for _ in range(num_moves):
            available = [m for m in all_moves if m[0] != last_face]
            move = random.choice(available)
            moves_applied.append(move)
            last_face = move[0]
        
        self.execute_moves(moves_applied)
        return moves_applied
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def __repr__(self) -> str:
        def face_str(f, row):
            return ' '.join(self.COLOR_NAMES[c] for c in self.state[f, row])
        
        lines = []
        
        for row in range(3):
            lines.append(f"       {face_str(0, row)}")
        
        lines.append("")
        
        for row in range(3):
            l = face_str(4, row)
            f = face_str(2, row)
            r = face_str(5, row)
            b = face_str(3, row)
            lines.append(f"{l}  {f}  {r}  {b}")
        
        lines.append("")
        
        for row in range(3):
            lines.append(f"       {face_str(1, row)}")
        
        return '\n'.join(lines)


def create_scrambled_cube(num_moves: int = 20, seed: Optional[int] = None) -> Tuple[RubiksCube, List[str]]:
    """Create a scrambled cube."""
    cube = RubiksCube()
    moves = cube.scramble(num_moves, seed)
    return cube, moves


if __name__ == "__main__":
    cube = RubiksCube()
    print("Solved:")
    print(cube)
    print()
    
    moves = cube.scramble(20, seed=42)
    print(f"Scramble: {' '.join(moves)}")
    print(cube)
    print(f"\nValid: {cube.is_valid()}")