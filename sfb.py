import pandas as pd
import argparse
import ast
from collections import Counter

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

def run_analysis(file_path: str):
    summary_fast, summary, best, model_score, top_final, bad_models = analyze_ensembles(file_path)

    print("\nNAJLEPSZY ENSEMBLE:")
    print(best)

    print("\nTOP 10 ENSEMBLES (fast):")
    print(summary_fast.head(10))
    
    print("\nTOP 10 ENSEMBLES:")
    print(summary.head(10))

    print("\nMODEL SCORE (top):")
    print(model_score.most_common(20))

    print("\final top:")
    print(top_final)

    print("\nBad models:")
    print(bad_models)
    
def safe_parse(ens):
    if pd.isna(ens):
        return []
    try:
        return list(ast.literal_eval(ens))
    except Exception:
        return []  # albo logowanie błędu
    
def analyze_ensembles(file_path: str):
    df = pd.read_csv(file_path)

    summary_fast = (
        df.groupby("ensemble", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            count=("accuracy", "count")
        )
        .sort_values(by="mean_accuracy", ascending=False)
    )


    print("summary done")
    best_ensemble = summary_fast.iloc[0]
    print("best chosen")
    top_ensembles_raw = summary_fast.head(10)["ensemble"]
    df_top = df[df["ensemble"].isin(top_ensembles_raw)].copy()
    

    df_top["ensemble_norm"] = df_top["ensemble"].apply(
        lambda x: tuple(sorted(safe_parse(x)))
    )

    summary = (
        df_top.groupby("ensemble_norm", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_confidence=("confidence", "mean"),
            count=("accuracy", "count")
        )
        .sort_values(by="mean_accuracy", ascending=False)
    )
    top_final = summary.head(20)
    
    model_score = Counter()

    for ens in top_final["ensemble_norm"]:
        model_score.update(ens)

    ensemble_sizes = [len(ens) for ens in top_final["ensemble_norm"]]
    
    all_models = set()
    for ens in df_top["ensemble_norm"]:
        all_models.update(ens)

    bad_models = all_models - set(model_score.keys())
    
    return summary_fast, summary, best_ensemble, model_score, top_final, bad_models

def add_ensemble_length_column(path="results.csv"):
    df = pd.read_csv(path)

    df["ensemble_size"] = df["ensemble"].apply(
        lambda x: len(x.split("+")) if isinstance(x, str) else 0
    )

    df.to_csv(path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Ścieżka do pliku CSV"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.file)
