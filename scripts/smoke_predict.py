import os
import requests
import pandas as pd

API = os.getenv("API_URL", "http://127.0.0.1:8000")


def main():
    # 1) Health
    r = requests.get(f"{API}/health", timeout=5)
    print("GET /health:", r.status_code, r.json())

    # 2) Charge 1–3 lignes du train et enlève la cible
    df = pd.read_csv("data/train.csv")
    X = df.drop(columns=["a_quitte_l_entreprise"])

    # 3) /predict (single)
    row = X.iloc[0].to_dict()
    r1 = requests.post(f"{API}/predict", json={"features": row}, timeout=10)
    print("POST /predict:", r1.status_code, r1.json())

    # 4) /predict_batch (batch 3 lignes)
    rows = X.iloc[:3].to_dict(orient="records")
    r2 = requests.post(f"{API}/predict_batch", json={"rows": rows}, timeout=10)
    print("POST /predict_batch:", r2.status_code, r2.json())


if __name__ == "__main__":
    main()
