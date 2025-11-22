
import os
import joblib
import pandas as pd
from typing import Optional

def save_results(results: pd.DataFrame, filename: str = "results_storage.pkl") -> None:
    """
    Save the prediction results DataFrame to a file using joblib (pickle format).

    The file is saved inside the 'result' directory, which should be mounted or
    mapped to a volume for persistence outside the container.

    Args:
        results: A pandas DataFrame containing the prediction results.
        filename: The name of the file to save the results as. Defaults to "results_storage.pkl".
    """
    # The path 'result' resolves to '/app/result' inside the Docker container
    
    filepath = os.path.join("result", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as f:
        joblib.dump(results, f)

    print(f"Results saved successfully in {filepath}")