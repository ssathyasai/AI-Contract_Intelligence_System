"""Task 6: Clause Understanding — same words, different order, different PE"""
import pickle, os
import numpy as np

def run(artifacts_dir="artifacts"):
    with open(f"{artifacts_dir}/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    contract_A = "Payment shall be made within 30 days."
    contract_B = "Within 30 days payment shall be made."

    def tok(text):
        return text.lower().replace('.', '').split()

    tA, tB = tok(contract_A), tok(contract_B)

    def positional_encoding(max_len=50, d_model=64):
        PE = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
                if i+1 < d_model:
                    PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
        return PE

    PE = positional_encoding()

    print("=" * 60)
    print("TASK 6 — Clause Understanding Analysis")
    print("=" * 60)
    print(f"Contract A: {contract_A}")
    print(f"Contract B: {contract_B}")
    print()
    print(f"{'Word':<12} {'A-pos':>6}  {'A-PE[:3]':>30}   {'B-pos':>6}  {'B-PE[:3]'}")
    print("-" * 80)
    for word in ["payment", "shall", "be", "made", "within", "30", "days"]:
        pa = tA.index(word) if word in tA else "N/A"
        pb = tB.index(word) if word in tB else "N/A"
        pe_a = PE[pa, :3].round(3) if isinstance(pa, int) else "—"
        pe_b = PE[pb, :3].round(3) if isinstance(pb, int) else "—"
        print(f"  {word:<12} {str(pa):>6}  {str(pe_a):>30}   {str(pb):>6}  {pe_b}")

    print("""
EXPLANATION:
  Both sentences use the EXACT SAME words.
  Positional Encoding assigns a unique vector to every position index.

  'payment' is at position 0 in A  →  PE[0] = [0, 1, 0, 1, ...]
  'payment' is at position 2 in B  →  PE[2] = [0.909, 0.416, ...]

  Final token representation = word_embedding + PE[position]
  So 'payment' produces a DIFFERENT final vector in A vs B.

  This is how the Transformer knows word ORDER, not just bag-of-words.
""")

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
