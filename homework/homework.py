# flake8: noqa: E501
import os
import gzip
import json
import pickle

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV


def preprocess_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns='ID', inplace=True)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.dropna(inplace=True)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df


def split_features_labels(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns="default")
    y_train = train_df["default"]
    X_test = test_df.drop(columns="default")
    y_test = test_df["default"]
    return X_train, y_train, X_test, y_test


def build_model_pipeline():
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numeric_features = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", StandardScaler(), numeric_features)
    ])

    return Pipeline([
        ("preprocessing", transformer),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("dim_reduction", PCA()),
        ("classifier", MLPClassifier(max_iter=15000, random_state=17)),
    ])


def define_grid_search(pipeline: Pipeline) -> GridSearchCV:
    param_grid = {
        'dim_reduction__n_components': [None],
        'feature_selection__k': [20],
        'classifier__hidden_layer_sizes': [(50, 30, 40, 60)],
        'classifier__alpha': [0.26],
        'classifier__learning_rate_init': [0.001],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )


def evaluate_model(model, X, y, dataset_name: str):
    predictions = model.predict(X)
    metrics = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": round(precision_score(y, predictions), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, predictions), 4),
        "recall": round(recall_score(y, predictions), 4),
        "f1_score": round(f1_score(y, predictions), 4)
    }
    return predictions, metrics


def create_confusion_report(y_true, y_pred, dataset_name: str):
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(matrix[0][0]), "predicted_1": int(matrix[0][1])},
        "true_1": {"predicted_0": int(matrix[1][0]), "predicted_1": int(matrix[1][1])}
    }


def save_pickle_gz(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(obj, f)


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def main():
    train_df = pd.read_csv("../files/input/train_data.csv.zip")
    test_df = pd.read_csv("../files/input/test_data.csv.zip")

    train_df = preprocess_credit_data(train_df)
    test_df = preprocess_credit_data(test_df)

    X_train, y_train, X_test, y_test = split_features_labels(train_df, test_df)

    base_pipeline = build_model_pipeline()
    search_model = define_grid_search(base_pipeline)
    search_model.fit(X_train, y_train)

    save_pickle_gz(search_model, '../files/models/model.pkl.gz')

    y_pred_train, train_metrics = evaluate_model(search_model, X_train, y_train, "train")
    y_pred_test, test_metrics = evaluate_model(search_model, X_test, y_test, "test")

    train_cm = create_confusion_report(y_train, y_pred_train, 'train')
    test_cm = create_confusion_report(y_test, y_pred_test, 'test')

    save_jsonl(
        [train_metrics, test_metrics, train_cm, test_cm],
        '../files/output/metrics.json'
    )


if __name__ == "__main__":
    main()
