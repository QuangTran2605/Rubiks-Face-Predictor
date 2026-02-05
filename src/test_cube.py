"""
test_cube.py - Verify cube simulator
"""

import numpy as np
from cube import RubiksCube


def run_tests():
    print("Running tests...\n")
    
    # Test 1: Solved state
    cube = RubiksCube()
    for i in range(6):
        assert np.all(cube.state[i] == i), f"Face {i} wrong"
    print("✓ Solved state correct")
    
    # Test 2: Move inverses cancel
    for move in ['U', 'D', 'F', 'B', 'L', 'R']:
        cube = RubiksCube()
        original = cube.state.copy()
        cube.execute_move(move)
        cube.execute_move(move + "'")
        assert np.array_equal(cube.state, original), f"{move} inverse failed"
    print("✓ Move inverses work")
    
    # Test 3: 4 moves = identity
    for move in ['U', 'D', 'F', 'B', 'L', 'R']:
        cube = RubiksCube()
        original = cube.state.copy()
        for _ in range(4):
            cube.execute_move(move)
        assert np.array_equal(cube.state, original), f"4x {move} failed"
    print("✓ Four moves = identity")
    
    # Test 4: Scrambling preserves colors
    for i in range(10):
        cube = RubiksCube()
        cube.scramble(50, seed=i)
        assert cube.is_valid(), f"Scramble {i} invalid"
    print("✓ Scrambling preserves color counts")
    
    # Test 5: Reproducibility
    cube1 = RubiksCube()
    cube1.scramble(20, seed=12345)
    cube2 = RubiksCube()
    cube2.scramble(20, seed=12345)
    assert np.array_equal(cube1.state, cube2.state), "Seeds not reproducible"
    print("✓ Same seed = same result")
    
    print("\n" + "="*40)
    print("ALL TESTS PASSED!")
    print("="*40)
    
    # Visual demo
    print("\nVisual demo:")
    cube = RubiksCube()
    cube.scramble(15, seed=42)
    print(cube)


if __name__ == "__main__":
    run_tests()