"""Task 4: Self-Attention Model — Embedding → MultiHeadAttention → Dense → Output"""
import numpy as np, pickle, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Embedding, Dense, Dropout, Input,
                                     LayerNormalization, MultiHeadAttention,
                                     GlobalAveragePooling1D)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

MAX_LEN   = 50
EMBED_DIM = 64

def build_attention_model(vocab_size, num_classes):
    inp   = Input(shape=(MAX_LEN,))
    x     = Embedding(vocab_size, EMBED_DIM)(inp)
    attn, _ = MultiHeadAttention(num_heads=4, key_dim=16)(x, x,
                                                           return_attention_scores=True)
    x     = LayerNormalization()(x + attn)
    x     = GlobalAveragePooling1D()(x)
    x     = Dense(64, activation="relu")(x)
    x     = Dropout(0.3)(x)
    out   = Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

def run(artifacts_dir="artifacts", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    sequences = np.load(f"{artifacts_dir}/sequences.npy")
    labels    = np.load(f"{artifacts_dir}/labels.npy")
    with open(f"{artifacts_dir}/word2idx.pkl",  "rb") as f: word2idx  = pickle.load(f)
    with open(f"{artifacts_dir}/label2idx.pkl", "rb") as f: label2idx = pickle.load(f)

    VOCAB_SIZE  = len(word2idx)
    NUM_CLASSES = len(label2idx)
    X_tr, X_te, y_tr, y_te = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels)

    model = build_attention_model(VOCAB_SIZE, NUM_CLASSES)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    model.fit(X_tr, y_tr, epochs=15, batch_size=32,
              validation_split=0.1, verbose=1)

    y_pred  = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print("TASK 4 — Attention Model Results")
    print("=" * 55)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    names = [k for k, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    print(classification_report(y_te, y_pred, target_names=names, zero_division=0))

    model.save(f"{models_dir}/attention_model.h5")
    print(f"Attention model saved → {models_dir}/attention_model.h5")
    return model

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
