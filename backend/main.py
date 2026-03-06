import io
import os
import warnings
import logging

import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sklearn import preprocessing
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Disease Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Resolve dataset paths relative to this file ───────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")

TRAIN_PATH = os.path.join(DATASETS_DIR, "pp5i_train.gr.csv")
CLASS_PATH = os.path.join(DATASETS_DIR, "pp5i_train_class.txt")
TEST_PATH  = os.path.join(DATASETS_DIR, "pp5i_test.gr.csv")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_bundled_train():
    train_df = pd.read_csv(TRAIN_PATH)
    class_df = pd.read_csv(CLASS_PATH)
    return train_df, class_df


def load_bundled_test():
    return pd.read_csv(TEST_PATH)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, class_df: pd.DataFrame):
    class_np    = class_df.to_numpy()
    le          = preprocessing.LabelEncoder()
    train_class = le.fit_transform(class_np.ravel())

    ttdf_sno = train_df["SNO"]
    ttdf_rem = train_df.iloc[:, 1:].clip(20, 16000)
    tsdf_sno = test_df["SNO"]
    tsdf_rem = test_df.iloc[:, 1:].clip(20, 16000)

    ttdf_cal = abs(ttdf_rem.max(axis=1) / ttdf_rem.min(axis=1))
    del_ind  = ttdf_cal[ttdf_cal < 2].index

    train_tdf = pd.concat([ttdf_sno.drop(del_ind), ttdf_rem.drop(del_ind)], axis=1, sort=False)
    test_tdf  = pd.concat([tsdf_sno.drop(del_ind), tsdf_rem.drop(del_ind)], axis=1, sort=False)

    f_vals, _ = f_classif(train_tdf.drop("SNO", axis=1).T, train_class)
    train_tdf["rank"] = f_vals
    test_tdf["rank"]  = f_vals
    train_tdf = train_tdf.sort_values("rank", ascending=False)
    test_tdf  = test_tdf.sort_values("rank", ascending=False)
    return train_tdf, test_tdf, train_class, le


def get_models(use_gs: bool):
    models = {
        "GaussianNB":   (GaussianNB(), {}),
        "DecisionTree": (DecisionTreeClassifier(random_state=42),
                         {"classifier__max_depth": [None, 10], "classifier__min_samples_split": [2, 5]}),
        "KNN":          (KNeighborsClassifier(),
                         {"classifier__n_neighbors": [3, 5, 7], "classifier__weights": ["uniform", "distance"]}),
        "MLP":          (MLPClassifier(random_state=42, max_iter=300),
                         {"classifier__hidden_layer_sizes": [(25, 25), (50, 50)],
                          "classifier__activation": ["relu", "tanh"],
                          "classifier__solver": ["sgd", "adam"]}),
        "ExtraTrees":   (ExtraTreesClassifier(random_state=42),
                         {"classifier__n_estimators": [100, 350], "classifier__max_depth": [None, 10]}),
        "RandomForest": (RandomForestClassifier(random_state=42),
                         {"classifier__n_estimators": [100, 300], "classifier__max_depth": [None, 10]}),
    }
    if not use_gs:
        return {k: (v[0], {}) for k, v in models.items()}
    return models


def run_evaluation(train_tdf, train_class, n_list, cv_folds, use_gs):
    models_dict = get_models(use_gs)
    model_names = list(models_dict.keys())
    error_rates = np.zeros((len(n_list), len(model_names)))
    results_rows = []

    full_x_train = train_tdf.drop(["SNO", "rank"], axis=1).to_numpy().T
    best_score, best_N, best_model_name, best_clf = 0, 0, "", None

    scoring = {
        "accuracy":  make_scorer(accuracy_score),
        "f1":        make_scorer(f1_score,        average="weighted", zero_division=0),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
    }

    for i, N in enumerate(n_list):
        x_trainN = full_x_train[:, :N]
        for j, model_name in enumerate(model_names):
            base_model, param_grid = models_dict[model_name]
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", base_model)])
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            if param_grid and use_gs:
                gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
                gs.fit(x_trainN, train_class)
                best_pipeline = gs.best_estimator_
                best_params   = gs.best_params_
            else:
                pipeline.fit(x_trainN, train_class)
                best_pipeline = pipeline
                best_params   = {}

            scores = cross_validate(best_pipeline, x_trainN, train_class, cv=cv, scoring=scoring)
            acc  = float(np.mean(scores["test_accuracy"]))
            f1   = float(np.mean(scores["test_f1"]))
            prec = float(np.mean(scores["test_precision"]))
            rec  = float(np.mean(scores["test_recall"]))

            error_rates[i, j] = 1 - acc
            results_rows.append({
                "n_genes": N, "model": model_name,
                "accuracy": round(acc, 4), "f1": round(f1, 4),
                "precision": round(prec, 4), "recall": round(rec, 4),
                "best_params": str(best_params),
            })

            if acc > best_score:
                best_score, best_N, best_model_name, best_clf = acc, N, model_name, best_pipeline

    return error_rates, model_names, results_rows, best_N, best_model_name, best_clf, best_score


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    n_list:          str        = Form("10,15,20,25,30"),
    cv_folds:        int        = Form(5),
    use_grid_search: bool       = Form(False),
    test_file:       UploadFile = File(None),
):
    # Parse n_list
    try:
        n_list_parsed = sorted([int(x.strip()) for x in n_list.split(",") if x.strip()])
        if not n_list_parsed:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=422, detail="n_list must be comma-separated integers, e.g. '10,15,20'")

    # Load training data (always bundled)
    try:
        train_df, class_df = load_bundled_train()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Bundled training data not found: {e}")

    # Load test data (uploaded or bundled)
    if test_file is not None and test_file.filename:
        contents = await test_file.read()
        test_df = pd.read_csv(io.BytesIO(contents))
    else:
        try:
            test_df = load_bundled_test()
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Bundled test data not found: {e}")

    # Preprocess
    try:
        train_tdf, test_tdf, train_class, le = preprocess(train_df, test_df, class_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    # Class distribution
    classes, counts = np.unique(train_class, return_counts=True)
    class_labels    = le.inverse_transform(classes).tolist()
    class_dist      = {label: int(cnt) for label, cnt in zip(class_labels, counts)}

    # Run evaluation
    try:
        error_rates, model_names, results_rows, best_N, best_model_name, best_clf, best_score = \
            run_evaluation(train_tdf, train_class, n_list_parsed, cv_folds, use_grid_search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {e}")

    # Best model row for F1
    best_f1 = next(
        (r["f1"] for r in results_rows if r["model"] == best_model_name and r["n_genes"] == best_N),
        0.0
    )

    # Predictions on test set
    x_test_full  = test_tdf.drop(["SNO", "rank"], axis=1).to_numpy().T
    x_test_bestN = x_test_full[:, :best_N]
    preds        = best_clf.predict(x_test_bestN)
    labels       = le.inverse_transform(preds.astype(int)).tolist()

    pred_counts_series = pd.Series(labels).value_counts()
    pred_counts = [{"disease": d, "count": int(c)} for d, c in pred_counts_series.items()]

    predictions = [{"patient": f"P{i+1}", "disease": lbl} for i, lbl in enumerate(labels)]

    return JSONResponse({
        "best_model":   best_model_name,
        "best_n":       best_N,
        "best_score":   round(best_score, 4),
        "best_f1":      round(best_f1, 4),
        "class_dist":   class_dist,
        "model_names":  model_names,
        "n_list":       n_list_parsed,
        "error_rates":  error_rates.tolist(),
        "results":      results_rows,
        "predictions":  predictions,
        "pred_counts":  pred_counts,
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
