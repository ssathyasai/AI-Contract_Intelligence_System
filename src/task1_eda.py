"""Task 1: EDA on Contract Intelligence Dataset"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import re, os

def run(data_path="data/contract_intelligence_500.csv", plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    print("=" * 55)
    print("TASK 1 — EDA: Contract Intelligence")
    print("=" * 55)
    print(f"Total contracts      : {len(df)}")
    print(f"\nClause distribution:\n{df['clause_type'].value_counts().to_string()}")

    df["length"] = df["contract_text"].astype(str).apply(len)
    print(f"\nAvg contract length  : {df['length'].mean():.1f} chars")
    print(f"Longest contract     : {df['length'].max()} chars")
    print(f"Shortest contract    : {df['length'].min()} chars")

    # ── Plot 1: Clause distribution ──────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    df["clause_type"].value_counts().plot(kind="bar", ax=ax,
                                          color="steelblue", edgecolor="black")
    ax.set_title("Clause Type Distribution")
    ax.set_xlabel("Clause Type"); ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(f"{plots_dir}/clause_distribution.png"); plt.close()

    # ── Plot 2: Word frequency ────────────────────────────────
    all_text = " ".join(df["contract_text"].astype(str)).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    stop  = {"this","that","with","from","shall","upon","have","been",
              "their","both","also","such","which","will","when","they"}
    words = [w for w in words if w not in stop]
    top20 = Counter(words).most_common(20)
    wlabels, wcounts = zip(*top20)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(wlabels, wcounts, color="coral", edgecolor="black")
    ax.set_title("Top 20 Word Frequencies")
    ax.set_xlabel("Word"); ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(f"{plots_dir}/word_frequency.png"); plt.close()

    # ── Plot 3: Contract length histogram ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["length"], bins=20, color="mediumseagreen", edgecolor="black")
    ax.set_title("Contract Length Distribution")
    ax.set_xlabel("Character Length"); ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/contract_length_hist.png"); plt.close()

    print(f"\nPlots saved → {plots_dir}/")
    return df

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
