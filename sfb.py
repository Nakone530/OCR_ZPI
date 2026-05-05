import pandas as pd

def run_analysis():
    file_path = input("Podaj ścieżkę do pliku CSV: ").strip()

    summary, best = analyze_ensembles(file_path)

    print("\nNAJLEPSZY ENSEMBLE:")
    print(best)

    print("\nTOP 10 ENSEMBLES:")
    print(summary.head(10))

def analyze_ensembles(file_path: str):
    df = pd.read_csv(file_path)

    # normalizacja ensemble (ważne przy tuple/string)
    df["ensemble"] = df["ensemble"].astype(str)

    summary = (
        df.groupby("ensemble")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_confidence=("confidence", "mean"),
            count=("accuracy", "count")
        )
        .sort_values("mean_accuracy", ascending=False)
    )

    best_ensemble = summary.iloc[0]

    return summary, best_ensemble

if __name__ == "__main__":
    run_analysis()
