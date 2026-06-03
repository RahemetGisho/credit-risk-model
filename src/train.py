# =========================
# Imports
# =========================

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

from src.data_processing import (
    load_data,
    process_data
)


# =========================
# Evaluation Function
# =========================

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions
        ),
        "recall": recall_score(
            y_test,
            predictions
        ),
        "f1": f1_score(
            y_test,
            predictions
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }


# =========================
# Main
# =========================

def main():

    # -------------------------
    # Load Data
    # -------------------------

    raw_df = load_data(
        "data/raw/data.csv"
    )

    X, full_df = process_data(raw_df)

    y = full_df["is_high_risk"]

    print("\nTarget Distribution")
    print(y.value_counts())

    print("\nTarget Distribution (%)")
    print(
        y.value_counts(
            normalize=True
        ) * 100
    )

    # -------------------------
    # Train Test Split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # MLflow
    # -------------------------


    mlflow.set_tracking_uri(
    "file:///C:/Users/user/Downloads/week4/credit-risk-model/mlruns")

    mlflow.set_experiment("credit-risk-model")

    # -------------------------
    # Baseline Models
    # -------------------------

    models = {
        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "random_forest":
            RandomForestClassifier(
                random_state=42
            ),

        "gradient_boosting":
            GradientBoostingClassifier(
                random_state=42
            )
    }

    best_baseline_model = None
    best_baseline_auc = 0

    # -------------------------
    # Train Baselines
    # -------------------------

    for name, model in models.items():

        with mlflow.start_run(
            run_name=name
        ):

            model.fit(
                X_train,
                y_train
            )

            metrics = evaluate(
                model,
                X_test,
                y_test
            )

            mlflow.log_params(
                model.get_params()
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.sklearn.log_model(
                model,
                name="model"
            )

            print(f"\n{name}")
            print(metrics)

            if metrics["roc_auc"] > best_baseline_auc:

                best_baseline_auc = metrics["roc_auc"]

                best_baseline_model = model

    # ==================================================
    # Grid Search - Random Forest
    # ==================================================

    rf_param_grid = {
        "n_estimators": [
            50,
            100,
            200
        ],
        "max_depth": [
            5,
            10,
            None
        ]
    }

    rf_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid=rf_param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    rf_search.fit(
        X_train,
        y_train
    )

    rf_tuned = rf_search.best_estimator_

    print("\nBest RF Parameters")
    print(
        rf_search.best_params_
    )

    rf_metrics = evaluate(
        rf_tuned,
        X_test,
        y_test
    )

    print("\nTuned Random Forest")
    print(rf_metrics)

    with mlflow.start_run(
        run_name="rf_grid_search"
    ):

        mlflow.log_params(
            rf_search.best_params_
        )

        mlflow.log_metrics(
            rf_metrics
        )

        mlflow.sklearn.log_model(
            rf_tuned,
            name="rf_tuned_model"
        )

    # ==================================================
    # Random Search - Gradient Boosting
    # ==================================================

    gb_param_dist = {
        "n_estimators": [
            50,
            100,
            150,
            200
        ],
        "learning_rate": [
            0.01,
            0.05,
            0.1,
            0.2
        ],
        "max_depth": [
            2,
            3,
            4,
            5
        ]
    }

    gb_search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(
            random_state=42
        ),
        param_distributions=gb_param_dist,
        n_iter=10,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )

    gb_search.fit(
        X_train,
        y_train
    )

    gb_tuned = gb_search.best_estimator_

    print("\nBest GB Parameters")
    print(
        gb_search.best_params_
    )

    gb_metrics = evaluate(
        gb_tuned,
        X_test,
        y_test
    )

    print("\nTuned Gradient Boosting")
    print(gb_metrics)

    with mlflow.start_run(
        run_name="gb_random_search"
    ):

        mlflow.log_params(
            gb_search.best_params_
        )

        mlflow.log_metrics(
            gb_metrics
        )

        mlflow.sklearn.log_model(
            gb_tuned,
            name="gb_tuned_model"
        )

    # ==================================================
    # Select Best Tuned Model
    # ==================================================

    candidate_models = {
        "random_forest_tuned": (
            rf_tuned,
            rf_metrics["roc_auc"]
        ),
        "gradient_boosting_tuned": (
            gb_tuned,
            gb_metrics["roc_auc"]
        )
    }

    final_name = max(
        candidate_models,
        key=lambda x:
        candidate_models[x][1]
    )

    final_model = (
        candidate_models[final_name][0]
    )

    print(
        f"\nFinal Selected Model: {final_name}"
    )

    # ==================================================
    # Register Best Model
    # ==================================================

    with mlflow.start_run(
        run_name="final_best_model"
    ):

        mlflow.sklearn.log_model(
            final_model,
            name="best_model",
            registered_model_name=
            "CreditRiskModel"
        )

    print(
        "\nBest model registered in MLflow"
    )


# =========================
# Run
# =========================

if __name__ == "__main__":
    main()