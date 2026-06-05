"""Task 2: Text Engineering — clean, tokenize, vocab, pad, OOV"""
import pandas as pd
import numpy as np
import re, pickle, os
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 2000
MAX_LEN    = 50

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def build_vocab(token_lists, size=VOCAB_SIZE):
    counter = Counter(t for toks in token_lists for t in toks)
    vocab = {"<PAD>": 0, "<OOV>": 1}
    for w, _ in counter.most_common(size - 2):
        vocab[w] = len(vocab)
    return vocab

def encode(tokens, vocab):
    return [vocab.get(t, 1) for t in tokens]

def run(data_path="data/contract_intelligence_500.csv",
        artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    df["clean_text"] = df["contract_text"].apply(clean)
    df["tokens"]     = df["clean_text"].apply(str.split)

    word2idx = build_vocab(df["tokens"])
    df["encoded"] = df["tokens"].apply(lambda t: encode(t, word2idx))

    sequences = pad_sequences(df["encoded"].tolist(),
                              maxlen=MAX_LEN, padding="post", truncating="post")

    label2idx = {l: i for i, l in enumerate(sorted(df["clause_type"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}
    labels    = np.array([label2idx[c] for c in df["clause_type"]])

    np.save(f"{artifacts_dir}/sequences.npy",  sequences)
    np.save(f"{artifacts_dir}/labels.npy",     labels)
    for name, obj in [("word2idx", word2idx),
                      ("label2idx", label2idx),
                      ("idx2label", idx2label)]:
        with open(f"{artifacts_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    print("=" * 55)
    print("TASK 2 — Text Engineering")
    print("=" * 55)
    print(f"Vocabulary size : {len(word2idx)}")
    print(f"Sequence shape  : {sequences.shape}")
    print(f"Labels          : {label2idx}")
    print("\nJustifications:")
    print("  Padding  → neural nets need fixed-length input; short seqs get 0-padded, long truncated.")
    print("  Vocab sz → 2000 covers all domain words; larger = memory cost; smaller = more OOV (<OOV>=1).")
    return sequences, labels, word2idx, label2idx, idx2label

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
