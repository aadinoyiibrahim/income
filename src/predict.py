import joblib
import pandas as pd
import os
import argparse
from preprocessing import load_and_preprocess

parser = argparse.ArgumentParser(description="Run predictions using the model.")
parser.add_argument(
    "--input",
    type=str,
    default=os.path.abspath("./data/2016-09-19_79351_training_july.csv"),
)
parser.add_argument("--model_dir", type=str, default=os.path.abspath("./models"))
parser.add_argument(
    "--output", type=str, default=os.path.abspath("./result/predictions_all_models.csv")
)
args = parser.parse_args()

print(f'path: {os.path.abspath("./models")}')


def process_predit(new_data_dir, model_dir, output_csv_path):
    """
    Load preprocessing artifacts, new data, and make predictions using
    all models in the model directory.
    """
    new_data_dir = os.path.abspath(new_data_dir)
    model_dir = os.path.abspath(model_dir)
    output_csv_path = os.path.abspath(output_csv_path)

    print("Loading preprocessing artifacts...")
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))

    print("Loading new data...")
    new_data = load_and_preprocess(new_data_dir, training=False)

    new_data["actual"] = new_data["direction"]

    model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.pkl")]

    all_predictions = {}

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)

        model_name = model_file.replace("_model.pkl", "")

        print(f"Making predictions with {model_name}...")

        class_predictions = model.predict(
            new_data.drop(columns=["actual", "direction", "user_id"], errors="ignore")
        )
        all_predictions[f"{model_name}_prediction"] = class_predictions

        # Add meaning
        pred_meaning = {0: "In", 1: "Out"}
        all_predictions[f"{model_name}_meaning"] = [
            pred_meaning[pred] for pred in class_predictions
        ]

        if model.__class__.__name__ in ["RandomForestClassifier", "MLPClassifier"]:
            y_pred_proba_test = model.predict_proba(
                new_data.drop(
                    columns=["actual", "direction", "user_id"], errors="ignore"
                )
            )[:, 1]
        elif model.__class__.__name__ == "LogisticRegressionModels":
            y_pred_proba_test = model.predict_proba(
                new_data.drop(
                    columns=["actual", "direction", "user_id"], errors="ignore"
                )
            )
        else:
            y_pred_proba_test = None

        if y_pred_proba_test is not None:
            all_predictions[f"{model_name}_probability"] = y_pred_proba_test

    output_df = pd.DataFrame(all_predictions)
    # output_df.insert(0, "user_id", new_data["user_id"])  # Add user_id
    output_df.insert(0, "actual", new_data["actual"])  # Add actual direction

    if output_csv_path is not None:
        output_df.to_csv(output_csv_path, index=False)  # Save CSV without index

    print(f"Predictions saved to {output_csv_path}")


if __name__ == "__main__":
    process_predit(args.input, args.model_dir, args.output)
