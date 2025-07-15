import os
import json
import pickle
import gzip
import glob
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    make_scorer
from sklearn.compose import ColumnTransformer


def load_train_test_data():
    train_df = pd.read_csv("../files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("../files/input/test_data.csv.zip", compression="zip")
    return train_df, test_df


def clean_dataset(df):
    df = df.rename(columns={'default payment next month': 'default'})
    df.drop(columns=['ID'], inplace=True)
    df = df[(df['MARRIAGE'] != 0) & (df['EDUCATION'] != 0)]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    return df.dropna()


def split_features_labels(df):
    return df.drop(columns=['default']), df['default']


def build_classification_pipeline(feature_columns):
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical = list(set(feature_columns) - set(categorical))

    preprocessing = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', MinMaxScaler(), numerical),
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('pca', PCA()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('mlp', MLPClassifier(random_state=12345, max_iter=1000))
    ])

    return pipeline


def perform_grid_search(pipeline, x_train, y_train):
    param_grid = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "feature_selection__k": [12],
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__alpha": [0.0001, 0.001],
    }

    scorer = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=10)

    grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search


def calculate_evaluation_metrics(dataset_name, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def generate_confusion_dict(dataset_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }


def save_model_to_gzip(estimator, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(estimator, f)


def write_metrics_to_json(metrics_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metrics in metrics_list:
            f.write(json.dumps(metrics) + "\n")


def run_model_pipeline():
    train_df, test_df = load_train_test_data()
    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    x_train, y_train = split_features_labels(train_df)
    x_test, y_test = split_features_labels(test_df)

    pipeline = build_classification_pipeline(x_train.columns)
    model = perform_grid_search(pipeline, x_train, y_train)

    save_model_to_gzip(model, "../files/models/model.pkl.gz")

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        calculate_evaluation_metrics("train", y_train, y_train_pred),
        calculate_evaluation_metrics("test", y_test, y_test_pred),
        generate_confusion_dict("train", y_train, y_train_pred),
        generate_confusion_dict("test", y_test, y_test_pred),
    ]

    write_metrics_to_json(metrics, "../files/output/metrics.json")


if __name__ == "__main__":
    run_model_pipeline()
