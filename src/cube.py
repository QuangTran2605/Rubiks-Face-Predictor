import copy
import random

"""
Rubik's cube representation and manipulation module, with realistic moves
"""
#Colors names
COLORS = {
    0: 'W',  
    1: 'Y', 
    2: 'R', 
    3: 'O', 
    4: 'B', 
    5: 'G'
}

FULL_COLORS = {
    0: 'White',  
    1: 'Yellow', 
    2: 'Red', 
    3: 'Orange', 
    4: 'Blue', 
    5: 'Green'
}

class RubiksCube:
    """
    Represents a Rubik's cube with 6 faces, each face having 9 stickers.
    Each sticker is represented by an integer from 0 to 5, corresponding to the color.
    The faces are ordered as: Up, Down, Left, Right, Front, Back.

    Face layout: 
                U U U
                U U U
                U U U
        L L L   F F F   R R R   B B B
        L L L   F F F   R R R   B B B
        L L L   F F F   R R R   B B B
                D D D
                D D D
                D D D
    Index each face as: 
        [0][0] [0][1] [0][2]
        [1][0] [1][1] [1][2]
        [2][0] [2][1] [2][2]
    """
    def __init__(self):
        """Create a solved cube."""
        self.state = {
            'U': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # white
            'D': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # yellow
            'F': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # red
            'B': [[3, 3, 3], [3, 3, 3], [3, 3, 3]],  # orange
            'L': [[4, 4, 4], [4, 4, 4], [4, 4, 4]],  # blue
            'R': [[5, 5, 5], [5, 5, 5], [5, 5, 5]],  # green
        }
     
    def _rotate_face_clockwise(self, face_name):
        """Rotate a face of the cube clockwise"""
        face = self.state[face_name]
        # Transpose and reverse rows
        self.state[face_name] = [
            [face[2][0], face[1][0], face[0][0]],
            [face[2][1], face[1][1], face[0][1]],
            [face[2][2], face[1][2], face[0][2]]
        ]
    
    def rotate_face_counterclockwise(self, face_name):
        """Rotate a face 90 degrees counter-clockwise."""
        # Three clockwise = one counter-clockwise
        for _ in range(3):
            self._rotate_face_clockwise(face_name)
    
    def move_U(self): 
        """Rotate the Up face clockwise."""
        self._rotate_face_clockwise('U')
        # Save front row
        temp = self.state['F'][0].copy()
        # F <- R
        self.state['F'][0] = self.state['R'][0].copy()
        # R <- B
        self.state['R'][0] = self.state['B'][0].copy()
        # B <- L
        self.state['B'][0] = self.state['L'][0].copy()
        # L <- temp (F)
        self.state['L'][0] = temp
    
    def move_U_prime(self):
        """Rotate the Up face counter-clockwise."""
        for _ in range(3):
            self.move_U()
    
    def move_D(self):
        """Rotate the Down face clockwise."""
        self._rotate_face_clockwise('D')
        # Save front row
        temp = self.state['F'][2].copy()
        # F <- L
        self.state['F'][2] = self.state['L'][2].copy()
        # L <- B
        self.state['L'][2] = self.state['B'][2].copy()
        # B <- R
        self.state['B'][2] = self.state['R'][2].copy()
        # R <- temp (F)
        self.state['R'][2] = temp
    
    def move_D_prime(self):
        """Rotate the Down face counter-clockwise."""
        for _ in range(3):
            self.move_D()
            
    def move_L(self):
        """Rotate the Left face clockwise."""
        self._rotate_face_clockwise('L')
        # Save U left column
        temp = [self.state['U'][i][0] for i in range(3)]
        # U <- F
        for i in range(3):
            self.state['U'][i][0] = self.state['F'][i][0]
        # F <- D
        for i in range(3):
            self.state['F'][i][0] = self.state['D'][i][0]
        # D <- B
        for i in range(3):
            self.state['D'][i][0] = self.state['B'][2-i][2]
        # B <- temp (U)
        for i in range(3):
            self.state['B'][i][2] = temp[i]

    def move_L_prime(self):
        """Rotate the Left face counter-clockwise."""
        for _ in range(3):
            self.move_L()
    
    def move_R(self):
        """Rotate the Right face clockwise."""
        self._rotate_face_clockwise('R')
        # Save U right column
        temp = [self.state['U'][i][2] for i in range(3)]
        # U <- B
        for i in range(3):
            self.state['U'][i][2] = self.state['B'][2-i][0]
        # B <- D
        for i in range(3):
            self.state['B'][i][0] = self.state['D'][i][2]
        # D <- F
        for i in range(3):
            self.state['D'][i][2] = self.state['F'][i][2]
        # F <- temp (U)
        for i in range(3):
            self.state['F'][i][2] = temp[i]
    
    def move_R_prime(self):
        """Rotate the Right face counter-clockwise."""
        for _ in range(3):
            self.move_R()

    def move_F(self):
        """Rotate the Front face clockwise."""
        self._rotate_face_clockwise('F')
        # Save U bottom row
        temp = self.state['U'][2].copy()
        # U <- L
        for i in range(3):
            self.state['U'][2][i] = self.state['L'][2-i][2]
        # L <- D
        for i in range(3):
            self.state['L'][i][2] = self.state['D'][0][i]
        # D <- R
        for i in range(3):
            self.state['D'][0][i] = self.state['R'][2-i][0]
        # R <- temp (U)
        for i in range(3):
            self.state['R'][i][0] = temp[i]

    def move_F_prime(self):
        """Rotate the Front face counter-clockwise."""
        for _ in range(3):
            self.move_F()
    
    def move_B(self):
        """Rotate the Back face clockwise."""
        self._rotate_face_clockwise('B')
        # Save U top row
        temp = self.state['U'][0].copy()
        # U <- R
        for i in range(3):
            self.state['U'][0][i] = self.state['R'][i][2]
        # R <- D
        for i in range(3):
            self.state['R'][i][2] = self.state['D'][2][2-i]
        # D <- L
        for i in range(3):
            self.state['D'][2][i] = self.state['L'][i][0]
        # L <- temp (U)
        for i in range(3):
            self.state['L'][i][0] = temp[2-i]
    
    def move_B_prime(self):
        """Rotate the Back face counter-clockwise."""
        for _ in range(3):
            self.move_B()
    
    def do_move(self, move):
        """Perform a move on the cube."""
        moves = {
            'U': self.move_U,
            "U'": self.move_U_prime,
            'D': self.move_D,
            "D'": self.move_D_prime,
            'L': self.move_L,
            "L'": self.move_L_prime,
            'R': self.move_R,
            "R'": self.move_R_prime,
            'F': self.move_F,
            "F'": self.move_F_prime,
            'B': self.move_B,
            "B'": self.move_B_prime,
        }
        if move in moves:
            moves[move]()
        else:
            raise ValueError(f"Invalid move: {move}")
    
    def scramble(self, num_moves=20):
        """
        Randomly scramble the cube with a given number of moves.
        Args:
            num_moves (int): Number of random moves to apply.
        Returns:
            List of moves applied.
        """
        possible_moves = ['U', "U'", 'D', "D'", 'L', "L'", 'R', "R'", 'F', "F'", 'B', "B'"]
        moves_applied = []
        for _ in range(num_moves):
            move = random.choice(possible_moves)
            self.do_move(move)
            moves_applied.append(move)
        return moves_applied
    
    def get_face(self, face_name):
        """Get a single face by name."""
        return self.state[face_name]
    
    def get_hidden_face(self, hidden='D'):
        """
        Get the hidden face.
        
        Args:
            hidden: Which face is hidden ('U', 'D', 'F', 'B', 'L', or 'R')
                    Default is 'D' (Down)
        
        Returns:
            The 3x3 grid of the hidden face
        """
        return self.state[hidden]
    
    def get_visible_faces(self, hidden='D'):
        """
        Get all faces except the hidden one.
        
        Args:
            hidden: Which face is hidden ('U', 'D', 'F', 'B', 'L', or 'R')
                    Default is 'D' (Down)
        
        Returns:
            Dict of 5 visible faces
        """
        visible = {}
        for name, face in self.state.items():
            if name != hidden:
                visible[name] = face
        return visible
    
    def get_all_face_names(self):
        """Get list of all face names."""
        return ['U', 'D', 'F', 'B', 'L', 'R']
    
    def print_face(self, face_name):
        """Print one face nicely."""
        print(f"{face_name}:")
        face = self.state[face_name]
        for row in face:
            color_names = [COLORS[c] for c in row]
            print(f"  {' '.join(color_names)}")
    
    def print_all(self):
        """Print all faces."""
        for face_name in ['U', 'D', 'L', 'R', 'F', 'B']:
            self.print_face(face_name)
            print()

if __name__ == "__main__":
    import random as rand
    
    # Test solved cube
    print("=== SOLVED CUBE ===\n")
    cube = RubiksCube()
    cube.print_all()
    
    # Scramble
    print("=== SCRAMBLING WITH 20 MOVES ===\n")
    moves = cube.scramble(20)
    print(f"Moves: {' '.join(moves)}\n")
    cube.print_all()
    
    # Test with different hidden faces
    print("\n=== TESTING DIFFERENT HIDDEN FACES ===")
    
    all_faces = cube.get_all_face_names()
    
    for hidden_face in all_faces:
        visible = cube.get_visible_faces(hidden=hidden_face)
        hidden = cube.get_hidden_face(hidden=hidden_face)
        
        print(f"\nIf {hidden_face} is hidden:")
        print(f"  Visible faces: {list(visible.keys())}")
        print(f"  Hidden face ({hidden_face}): {hidden[0]} ...")  # Just first row
    
    # Example: Random hidden face (like in training)
    print("\n=== RANDOM HIDDEN FACE EXAMPLE ===")
    random_hidden = rand.choice(all_faces)
    print(f"Randomly selected hidden face: {random_hidden}")
    
    visible = cube.get_visible_faces(hidden=random_hidden)
    hidden = cube.get_hidden_face(hidden=random_hidden)
    
    print(f"\nVisible faces (INPUT to model):")
    for face_name, face in visible.items():
        print(f"  {face_name}: {face}")
    
    print(f"\nHidden face (OUTPUT - what model predicts):")
    print(f"  {random_hidden}: {hidden}")