"""
train.py — run ALL tasks in order.
Usage:  python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

BASE      = os.path.dirname(__file__)
DATA      = os.path.join(BASE, "data",      "contract_intelligence_500.csv")
ARTIFACTS = os.path.join(BASE, "artifacts")
MODELS    = os.path.join(BASE, "models")
PLOTS     = os.path.join(BASE, "plots")

for d in [ARTIFACTS, MODELS, PLOTS]:
    os.makedirs(d, exist_ok=True)

from src.task1_eda              import run as eda
from src.task2_text_engineering import run as text_eng
from src.task3_baseline_model   import run as baseline
from src.task4_attention_model  import run as attention
from src.task5_positional_encoding import run as pos_enc
from src.task6_clause_understanding import run as clause
from src.task7_attention_analysis   import run as attn_analysis

if __name__ == "__main__":
    print("\n[1/7] EDA")
    eda(DATA, PLOTS)

    print("\n[2/7] Text Engineering")
    text_eng(DATA, ARTIFACTS)

    print("\n[3/7] Baseline Model")
    baseline(ARTIFACTS, MODELS)

    print("\n[4/7] Attention Model")
    attention(ARTIFACTS, MODELS)

    print("\n[5/7] Positional Encoding")
    pos_enc(PLOTS)

    print("\n[6/7] Clause Understanding")
    clause(ARTIFACTS)

    print("\n[7/7] Attention Analysis")
    attn_analysis(ARTIFACTS, MODELS, PLOTS)

    print("\n✅  All tasks complete.")
    print("   Launch dashboard:  streamlit run app.py")
