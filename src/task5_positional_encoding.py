"""Task 5: Positional Encoding from scratch + heatmaps"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal PE from scratch (no library)."""
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2 * i / d_model)))
    return PE

def run(plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    MAX_LEN = 50
    D_MODEL = 64
    PE = positional_encoding(MAX_LEN, D_MODEL)

    print("=" * 55)
    print("TASK 5 — Positional Encoding")
    print("=" * 55)
    print(f"PE shape: {PE.shape}")
    for p in [0, 1, 2]:
        print(f"Position {p}: {PE[p, :6].round(4)} ...")

    # Full heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_title("Positional Encoding Heatmap — All Positions")
    ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Position")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/pe_full_heatmap.png"); plt.close()

    # Individual position slices
    for pos in [0, 1, 2, 5, 10, 20]:
        fig, ax = plt.subplots(figsize=(12, 1.8))
        ax.imshow(PE[pos:pos+1, :], aspect='auto', cmap='plasma')
        ax.set_title(f"Positional Encoding — Position {pos}")
        ax.set_xlabel("Encoding Dimension"); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/pe_pos_{pos}.png"); plt.close()
        print(f"  Saved: pe_pos_{pos}.png")

    print(f"Heatmaps saved → {plots_dir}/")
    return PE

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
