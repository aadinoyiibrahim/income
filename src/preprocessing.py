import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess(data_path, training=True, scaler=None):
    """
    Load and preprocess dataset.
    If training=True, fit a new scaler; otherwise, use provided scaler for new data.
    """
    mcc_dir = os.path.abspath("./data/mcc_group_definition.csv")
    trans_dir = os.path.abspath("./data/transaction_types.csv")
    df_train = pd.read_csv(data_path)
    print(f"original df shape: {df_train.shape}")
    df_mcc_group = pd.read_csv(mcc_dir)
    df_transaction = pd.read_csv(trans_dir)

    print(f"actual {training}")

    df_train = df_train.drop(["dataset_transaction", "dataset_user"], axis=1)

    # add --direction-- feature
    selected_transaction_types = list(df_train.transaction_type.unique())
    df_transaction_filtered = df_transaction[
        df_transaction.type.isin(selected_transaction_types)
    ]

    type_in = df_transaction_filtered[df_transaction_filtered.direction == "In"].type
    type_out = df_transaction_filtered[df_transaction_filtered.direction == "Out"].type
    type_in, type_out = list(type_in), list(type_out)

    df_train["direction"] = df_train.transaction_type.apply(
        lambda x: "In" if x in type_in else "Out"
    )

    # add --agent-- feature
    type_to_agent = {
        "BBU": "Partner",
        "CT": "Bank Account",
        "DR": "Bank Account",
        "PF": "Card",
        "PT": "Card",
        "BUB": "Partner",
        "DD": "Bank Account",
        "DT": "Bank Account",
        "FT": "Bank Account",
        "TUB": "Partner",
    }

    df_train["agent"] = df_train.transaction_type.apply(lambda x: type_to_agent[x])

    # feature engineering
    df_train["transaction_date"] = pd.to_datetime(df_train["transaction_date"])
    df_train["day"] = df_train["transaction_date"].dt.day
    df_train["month"] = df_train["transaction_date"].dt.month

    df_train = fill_missing_with_mode(df_train, "user_id", "mcc_group")

    df_train.drop(
        ["user_id", "transaction_date", "transaction_type"], axis=1, inplace=True
    )

    # Encode categorical features
    categorical_cols = df_train.select_dtypes(include=["object"]).columns
    print(f"categorical_cols:{categorical_cols}")
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df_train[col] = encoders[col].fit_transform(df_train[col])

    # Convert 'mcc_group' to integer
    df_train["mcc_group"] = df_train["mcc_group"].astype(int)

    features = ["mcc_group", "amount_n26_currency", "day", "month"]
    X = df_train.drop(columns=["direction"])
    y = df_train["direction"]
    print(f"y:{type(y)}")

    # data split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Standardizing the features
    if training:
        print(f"using --training == training--")
        scaler = StandardScaler()
        X_train[features] = scaler.fit_transform(X_train[features])
        X_val[features] = scaler.transform(X_val[features])
        X_test[features] = scaler.transform(X_test[features])

        print(f"train size: {X_train.shape[0] / X.shape[0] * 100:.2f}%")
        print(f"val size: {X_val.shape[0] / X.shape[0] * 100:.2f}%")
        print(f"test size: {X_test.shape[0] / X.shape[0] * 100:.2f}%")

        print(f"direction>y_train: {y_train.value_counts(normalize=True) * 100}")
        print(f"direction>y_val: {y_val.value_counts(normalize=True) * 100}")
        print(f"direction>y_test: {y_test.value_counts(normalize=True) * 100}")

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoders
    else:
        print(f"using --training == predict X--")
        return X


def fill_missing_with_mode(df, group_col, target_col):
    """
    Fill missing values with the local mode or misscelaneious value 16
    """

    def mode_function(x):
        modes = x.mode()
        if not modes.empty:
            return modes[0]
        else:
            return 16

    df[target_col] = df.groupby(group_col)[target_col].transform(
        lambda x: x.fillna(mode_function(x))
    )

    return df
