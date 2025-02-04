import os
import joblib


def save_results(results, filename="results_storage.pkl"):
    """
    Save results in a file.
    """
    filepath = os.path.join("result", filename)
    with open(filepath, "wb") as f:
        joblib.dump(results, f)

    print(f"Results saved successfully in {filepath}")
