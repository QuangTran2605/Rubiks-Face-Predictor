"""
visualize_solutions.py - Visualize combined solver results

Takes a random sample, runs the solver, and displays:
1. The 5 visible faces + hidden face layout
2. The correct solution with its neighbors from the solution set
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from combined_solver import (
    solve_hidden_face,
    check_solution,
    pattern_to_string,
    COLOR_NAMES,
    FACE_NAMES
)

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

# Color mapping for visualization (RGB values)
COLOR_RGB = {
    0: '#FFFFFF',  # White
    1: '#FFFF00',  # Yellow
    2: '#00FF00',  # Green
    3: '#0000FF',  # Blue
    4: '#FFA500',  # Orange
    5: '#FF0000',  # Red
    -1: '#808080', # Gray (hidden/unknown)
}

# Face layout for unfolded cube visualization
# Standard cross layout:
#       U
#     L F R B
#       D
FACE_POSITIONS = {
    U: (1, 0),  # row 0, col 1
    L: (0, 1),  # row 1, col 0
    F: (1, 1),  # row 1, col 1
    R: (2, 1),  # row 1, col 2
    B: (3, 1),  # row 1, col 3
    D: (1, 2),  # row 2, col 1
}


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_face(
    ax: plt.Axes,
    face_data: np.ndarray,
    x_offset: float,
    y_offset: float,
    cell_size: float = 1.0,
    title: str = "",
    highlight: bool = False,
    show_indices: bool = False,
    title_offset: float = 0.6
):
    """
    Draw a single face (3x3 grid) on the axes.
    
    Args:
        ax: Matplotlib axes
        face_data: (9,) array of colors or (3,3)
        x_offset: X position for top-left corner
        y_offset: Y position for top-left corner
        cell_size: Size of each sticker cell
        title: Title to display above face
        highlight: If True, draw border around face
        show_indices: If True, show sticker indices
    """
    if face_data.shape == (9,):
        face_data = face_data.reshape(3, 3)
    
    for row in range(3):
        for col in range(3):
            color_idx = face_data[row, col]
            color = COLOR_RGB.get(int(color_idx), '#808080')
            
            x = x_offset + col * cell_size
            y = y_offset + row * cell_size
            
            # Draw cell
            rect = patches.FancyBboxPatch(
                (x, y),
                cell_size * 0.95,
                cell_size * 0.95,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Show index if requested
            if show_indices:
                idx = row * 3 + col
                ax.text(
                    x + cell_size * 0.5,
                    y + cell_size * 0.5,
                    str(idx),
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='gray'
                )
    
    # Highlight border
    if highlight:
        border = patches.Rectangle(
            (x_offset - 0.05, y_offset - 0.05),
            3 * cell_size + 0.1,
            3 * cell_size + 0.1,
            fill=False,
            edgecolor='red',
            linewidth=3
        )
        ax.add_patch(border)
    
    # Title
    if title:
        ax.text(
            x_offset + 1.5 * cell_size,
            y_offset - title_offset,
            title,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )


def draw_unfolded_cube(
    ax: plt.Axes,
    faces: Dict[int, np.ndarray],
    hidden_face_idx: int,
    title: str = "Cube State",
    show_hidden: bool = True, 
    title_offset: float = 0.6
):
    """
    Draw unfolded cube in cross layout.
    
    Args:
        ax: Matplotlib axes
        faces: Dictionary mapping face index to (9,) array
        hidden_face_idx: Which face is hidden
        title: Overall title
        show_hidden: If True, show hidden face; if False, show as gray
    """
    cell_size = 1.0
    
    for face_idx, (col, row) in FACE_POSITIONS.items():
        x_offset = col * 3 * cell_size + col * 0.5
        y_offset = row * 3 * cell_size + row * 0.5
        
        is_hidden = (face_idx == hidden_face_idx)
        
        if is_hidden and not show_hidden:
            # Show as gray/unknown
            face_data = np.full(9, -1, dtype=np.int8)
        else:
            face_data = faces.get(face_idx, np.full(9, -1, dtype=np.int8))
        
        face_title = FACE_NAMES[face_idx]
        if is_hidden:
            face_title += " (HIDDEN)"
        
        draw_face(
            ax, face_data, x_offset, y_offset,
            cell_size=cell_size,
            title=face_title,
            highlight=is_hidden,
            title_offset=title_offset
        )
    
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)


def draw_solution_pattern(
    ax: plt.Axes,
    pattern: np.ndarray,
    title: str = "",
    highlight: bool = False,
    is_correct: bool = False
):
    """
    Draw a single solution pattern (hidden face).
    
    Args:
        ax: Matplotlib axes
        pattern: (9,) array
        title: Title for this pattern
        highlight: Draw highlight border
        is_correct: If True, show green border for correct solution
    """
    cell_size = 1.0
    
    draw_face(
        ax, pattern,
        x_offset=0, y_offset=0,
        cell_size=cell_size,
        title="",
        highlight=False
    )
    
    # Border for correct solution
    if is_correct:
        border = patches.Rectangle(
            (-0.1, -0.1),
            3 * cell_size + 0.2,
            3 * cell_size + 0.2,
            fill=False,
            edgecolor='green',
            linewidth=4
        )
        ax.add_patch(border)
        ax.text(
            1.5, 3.3,
            "✓ CORRECT",
            ha='center',
            fontsize=10,
            color='green',
            fontweight='bold'
        )
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=10)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_random_sample(
    data_split: str = 'test',
    sample_idx: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load a random sample from the dataset.
    
    Args:
        data_split: 'train', 'val', or 'test'
        sample_idx: Specific sample index (None for random)
        seed: Random seed
    
    Returns:
        (visible_faces, target, hidden_idx, sample_idx)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / f"{data_split}.npz")
    
    n = len(data['target'])
    
    if sample_idx is None:
        sample_idx = random.randint(0, n - 1)
    
    return (
        data['visible'][sample_idx],
        data['target'][sample_idx],
        int(data['hidden_idx'][sample_idx]),
        sample_idx
    )


def reconstruct_full_cube(
    visible_faces: np.ndarray,
    hidden_face_idx: int,
    hidden_pattern: Optional[np.ndarray] = None
) -> Dict[int, np.ndarray]:
    """
    Reconstruct full cube state from visible faces.
    
    Args:
        visible_faces: (5, 9) array
        hidden_face_idx: Which face is hidden
        hidden_pattern: Pattern for hidden face (or None)
    
    Returns:
        Dictionary mapping face index to (9,) array
    """
    faces = {}
    visible_idx = 0
    
    for face_idx in range(6):
        if face_idx == hidden_face_idx:
            if hidden_pattern is not None:
                faces[face_idx] = hidden_pattern
            else:
                faces[face_idx] = np.full(9, -1, dtype=np.int8)
        else:
            faces[face_idx] = visible_faces[visible_idx]
            visible_idx += 1
    
    return faces


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================

def get_neighbor_solutions(
    patterns: List[np.ndarray],
    correct_idx: int,
    num_neighbors: int = 2
) -> Tuple[List[np.ndarray], List[int], int]:
    """
    Get the correct solution and its neighbors from the solution list.
    Wraps around if at edges.
    
    Args:
        patterns: List of solution patterns
        correct_idx: Index of correct solution
        num_neighbors: Total number of neighbors to get (split left/right)
    
    Returns:
        (selected_patterns, selected_indices, correct_position_in_selection)
    
    Examples:
        patterns = [0,1,2,3,4,5,6,7,8,9], correct_idx = 0, num_neighbors = 2
        -> shows [9, 0, 1] (wraps around)
        
        patterns = [0,1,2,3,4,5,6,7,8,9], correct_idx = 9, num_neighbors = 2
        -> shows [8, 9, 0] (wraps around)
        
        patterns = [0,1,2,3,4,5,6,7,8,9], correct_idx = 5, num_neighbors = 2
        -> shows [4, 5, 6] (no wrap needed)
    """
    n = len(patterns)
    
    # If we have fewer patterns than needed, return all
    total_needed = num_neighbors + 1
    if n <= total_needed:
        return patterns, list(range(n)), correct_idx
    
    # Calculate neighbors on each side
    left_count = num_neighbors // 2
    right_count = num_neighbors - left_count
    
    # Build indices with wrap-around
    selected_indices = []
    
    # Left neighbors (going backwards, wrapping if needed)
    for i in range(left_count, 0, -1):
        idx = (correct_idx - i) % n  # Wrap around
        selected_indices.append(idx)
    
    # Correct solution
    selected_indices.append(correct_idx)
    
    # Right neighbors (going forwards, wrapping if needed)
    for i in range(1, right_count + 1):
        idx = (correct_idx + i) % n  # Wrap around
        selected_indices.append(idx)
    
    selected_patterns = [patterns[i] for i in selected_indices]
    correct_position = left_count  # Correct is always after left neighbors
    
    return selected_patterns, selected_indices, correct_position


def visualize_sample(
    data_split: str = 'test',
    sample_idx: Optional[int] = None,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Visualize a sample with its solutions.
    
    Args:
        data_split: 'train', 'val', or 'test'
        sample_idx: Specific sample (None for random)
        seed: Random seed
        save_path: If provided, save figure to this path
        show_plot: If True, display the plot
    """
    # Load sample
    visible, target, hidden_idx, idx = load_random_sample(
        data_split, sample_idx, seed
    )
    
    print("=" * 60)
    print(f"VISUALIZING SAMPLE {idx} from {data_split}")
    print("=" * 60)
    print(f"Hidden face: {FACE_NAMES[hidden_idx]}")
    
    # Run solver
    patterns = solve_hidden_face(visible, hidden_idx)
    print(f"Solutions found: {len(patterns)}")
    
    # Find correct solution
    found, correct_idx = check_solution(patterns, target)
    
    if not found:
        print("ERROR: Target not found in solutions!")
        return
    
    print(f"Correct solution at index: {correct_idx}")
    
    # Get neighbors
    selected_patterns, selected_indices, correct_pos = get_neighbor_solutions(
        patterns, correct_idx, num_neighbors=2
    )
    
    print(f"Showing solutions at indices: {selected_indices}")
    print(f"Correct solution position in selection: {correct_pos}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Grid: top row for cube, bottom row for solutions
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.5, 1],
        width_ratios=[1, 1, 1, 1],
        hspace=0.4,
        wspace=0.3
    )
    
    # =================================
    # Top: Cube visualization
    # =================================
    ax_cube = fig.add_subplot(gs[0, :])
    
    # Reconstruct cube with hidden face shown
    faces = reconstruct_full_cube(visible, hidden_idx, target)
    
    draw_unfolded_cube(
        ax_cube, faces, hidden_idx,
        title=f"Sample {idx} - Hidden Face: {FACE_NAMES[hidden_idx]} (shown in red border)",
        show_hidden=True
    )
    
    # =================================
    # Bottom: Solutions
    # =================================
    
    # Info text
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    
    info_text = f"""
    Sample: {idx}
    Dataset: {data_split}
    Hidden face: {FACE_NAMES[hidden_idx]}
    
    Total solutions: {len(patterns)}
    Correct at index: {correct_idx}
    
    Showing indices:
    {selected_indices}
    """
    
    ax_info.text(
        0.1, 0.9, info_text.strip(),
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Solution patterns
    for i, (pattern, sol_idx) in enumerate(zip(selected_patterns, selected_indices)):
        ax = fig.add_subplot(gs[1, i + 1])
        
        is_correct = (sol_idx == correct_idx)
        title = f"Solution #{sol_idx}"
        if is_correct:
            title += " (TARGET)"
        
        draw_solution_pattern(
            ax, pattern,
            title=title,
            is_correct=is_correct
        )
    
    # Main title
    fig.suptitle(
        "Cube Hidden Face Solver - Solution Visualization",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    # Add color legend
    add_color_legend(fig)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def add_color_legend(fig: plt.Figure):
    """Add a color legend to the figure."""
    legend_elements = [
        patches.Patch(facecolor=COLOR_RGB[i], edgecolor='black', label=COLOR_NAMES[i])
        for i in range(6)
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=6,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.98, 0.98)
    )


# =============================================================================
# BATCH VISUALIZATION
# =============================================================================

def visualize_multiple_samples(
    num_samples: int = 5,
    data_split: str = 'test',
    save_dir: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Visualize multiple random samples.
    
    Args:
        num_samples: Number of samples to visualize
        data_split: Dataset split to use
        save_dir: Directory to save images (None to skip saving)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / f"{data_split}.npz")
    n = len(data['target'])
    
    # Select random indices
    indices = random.sample(range(n), min(num_samples, n))
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(indices):
        print(f"\n{'=' * 60}")
        print(f"Sample {i + 1}/{num_samples}")
        
        save_file = None
        if save_dir:
            save_file = str(save_path / f"sample_{idx}.png")
        
        visualize_sample(
            data_split=data_split,
            sample_idx=idx,
            save_path=save_file,
            show_plot=False
        )
    
    print(f"\n{'=' * 60}")
    print(f"Generated {num_samples} visualizations")
    if save_dir:
        print(f"Saved to: {save_dir}")


# =============================================================================
# SOLUTION DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_solution_distribution(
    data_split: str = 'test',
    max_samples: int = 1000
):
    """
    Analyze the distribution of solution counts.
    """
    print("=" * 60)
    print("ANALYZING SOLUTION DISTRIBUTION")
    print("=" * 60)
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    data = np.load(data_dir / f"{data_split}.npz")
    
    n = min(len(data['target']), max_samples)
    
    solution_counts = []
    
    for i in range(n):
        patterns = solve_hidden_face(data['visible'][i], int(data['hidden_idx'][i]))
        solution_counts.append(len(patterns))
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n}...")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    counts, bins, _ = ax1.hist(
        solution_counts, 
        bins=range(min(solution_counts), max(solution_counts) + 2),
        edgecolor='black',
        alpha=0.7
    )
    ax1.set_xlabel('Number of Solutions')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Solution Count Distribution (n={n})')
    ax1.axvline(np.mean(solution_counts), color='red', linestyle='--', label=f'Mean: {np.mean(solution_counts):.2f}')
    ax1.axvline(np.median(solution_counts), color='green', linestyle='--', label=f'Median: {np.median(solution_counts):.1f}')
    ax1.legend()
    
    # Box plot by hidden face
    ax2 = axes[1]
    
    by_face = {i: [] for i in range(6)}
    for i in range(n):
        face = int(data['hidden_idx'][i])
        patterns = solve_hidden_face(data['visible'][i], face)
        by_face[face].append(len(patterns))
    
    box_data = [by_face[i] for i in range(6)]
    bp = ax2.boxplot(box_data, labels=FACE_NAMES, patch_artist=True)
    
    # Color boxes
    for patch, face_idx in zip(bp['boxes'], range(6)):
        patch.set_facecolor(COLOR_RGB[face_idx])
        if face_idx in [0, 1]:  # White/Yellow need dark text
            patch.set_alpha(0.7)
    
    ax2.set_xlabel('Hidden Face')
    ax2.set_ylabel('Number of Solutions')
    ax2.set_title('Solution Count by Hidden Face')
    
    plt.tight_layout()
    plt.savefig('solution_distribution.png', dpi=150)
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Min: {min(solution_counts)}")
    print(f"  Max: {max(solution_counts)}")
    print(f"  Mean: {np.mean(solution_counts):.2f}")
    print(f"  Median: {np.median(solution_counts):.1f}")
    print(f"  Std: {np.std(solution_counts):.2f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize cube solver solutions')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--idx', type=int, default=None,
                        help='Specific sample index (random if not provided)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figure')
    parser.add_argument('--batch', type=int, default=None,
                        help='Generate multiple samples')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze solution distribution')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_solution_distribution(args.split)
    elif args.batch:
        visualize_multiple_samples(
            num_samples=args.batch,
            data_split=args.split,
            save_dir='visualizations',
            seed=args.seed
        )
    else:
        print("\n" + "=" * 60)
        print("Here's an example of the solution:")
        print("=" * 60 + "\n")
        
        visualize_sample(
            data_split=args.split,
            sample_idx=args.idx,
            seed=args.seed,
            save_path=args.save,
            show_plot=True
        )