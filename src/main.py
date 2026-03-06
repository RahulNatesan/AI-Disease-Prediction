import argparse
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
logging.getLogger("sklearn").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def load_data(train_url, test_url, class_url):
    """Loads the training, testing and class datasets."""
    print(f"Loading data from:\n Train: {train_url}\n Test: {test_url}\n Class: {class_url}")
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    class_df = pd.read_csv(class_url)
    return train_df, test_df, class_df

def preprocess_data(train_df, test_df, class_df):
    """Preprocesses data: labels encoding, clipping, filtering and feature ranking."""
    print("Preprocessing data...")
    class_np = class_df.to_numpy()
    le = preprocessing.LabelEncoder()
    train_class = le.fit_transform(class_np.ravel())

    # Extract SNO and features
    ttdf_sno = train_df['SNO']
    ttdf_rem = train_df.iloc[:, 1:].clip(20, 16000)

    tsdf_sno = test_df['SNO']
    tsdf_rem = test_df.iloc[:, 1:].clip(20, 16000)

    # Filter out rows where max/minfold is < 2
    ttdf_cal = abs(ttdf_rem.max(axis=1) / ttdf_rem.min(axis=1))
    del_ind = ttdf_cal[ttdf_cal < 2].index

    train_tdf = pd.concat([ttdf_sno.drop(del_ind), ttdf_rem.drop(del_ind)], axis=1, sort=False)
    test_tdf = pd.concat([tsdf_sno.drop(del_ind), tsdf_rem.drop(del_ind)], axis=1, sort=False)

    # Feature selection ranking using ANOVA F-value
    tTrain_tdf = train_tdf.drop('SNO', axis=1).T
    f_values, _ = f_classif(tTrain_tdf, train_class)
    
    train_tdf['rank'] = f_values
    test_tdf['rank'] = f_values

    train_tdf = train_tdf.sort_values('rank', ascending=False)
    test_tdf = test_tdf.sort_values('rank', ascending=False)

    return train_tdf, test_tdf, train_class, le

def get_models():
    """Returns a dictionary of models and their hyperparameter grids for tuning."""
    models = {
        'GaussianNB': (GaussianNB(), {}),
        'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }),
        'KNeighborsClassifier': (KNeighborsClassifier(), {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance']
        }),
        'MLP': (MLPClassifier(random_state=42, max_iter=300), {
            'classifier__hidden_layer_sizes': [(25, 25), (50, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__solver': ['sgd', 'adam']
        }),
        'ExtraTreesClassifier': (ExtraTreesClassifier(random_state=42), {
            'classifier__n_estimators': [100, 350],
            'classifier__max_depth': [None, 10]
        }),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
            'classifier__n_estimators': [100, 300],
            'classifier__max_depth': [None, 10]
        })
    }
    return models

def evaluate_models_on_subsets(train_tdf, train_class, n_list, results_dir="results"):
    """Evaluates various models on different top-N feature subsets."""
    print("Evaluating models on different gene subsets...")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    models_dict = get_models()
    model_names = list(models_dict.keys())
    
    # Store error rates for heatmap: rows=N, cols=Models
    error_rates = np.zeros((len(n_list), len(model_names)))
    
    # Track the absolute best model
    best_overall_score = 0
    best_overall_model_name = ""
    best_overall_N = 0
    best_overall_clf = None

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    # Prepare features: Drop SNO and rank, convert to numpy, shape transpose for features x samples -> samples x features
    full_x_train = train_tdf.drop(['SNO', 'rank'], axis=1).to_numpy().T

    for i, N in enumerate(n_list):
        print(f"--- Evaluating Top {N} Genes ---")
        # take top N features. (features were sorted by rank descending earlier)
        # Note full_x_train is (samples, features) since we transposed
        x_trainN = full_x_train[:, :N]
        
        for j, model_name in enumerate(model_names):
            base_model, param_grid = models_dict[model_name]
            
            # Create a pipeline with a scaler
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', base_model)
            ])
            
            # Use GridSearchCV for hyperparameter tuning
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Fit to find best params
            grid_search.fit(x_trainN, train_class)
            best_model = grid_search.best_estimator_
            
            # Evaluate the best pipeline using multiple metrics
            scores = cross_validate(best_model, x_trainN, train_class, cv=cv, scoring=scoring)
            
            mean_acc = np.mean(scores['test_accuracy'])
            mean_f1 = np.mean(scores['test_f1'])
            error_rate = 1 - mean_acc
            
            error_rates[i, j] = error_rate
            
            print(f"  {model_name}: Acc={mean_acc:.4f}, F1={mean_f1:.4f} (Best Params: {grid_search.best_params_})")
            
            if mean_acc > best_overall_score:
                best_overall_score = mean_acc
                best_overall_model_name = model_name
                best_overall_N = N
                best_overall_clf = best_model

    plot_results(error_rates, n_list, model_names, results_dir)
    
    print("\n" + "="*40)
    print(f"Best Configuration Found:")
    print(f"Best N Features: {best_overall_N}")
    print(f"Best Classifier: {best_overall_model_name}")
    print(f"Best Accuracy  : {best_overall_score:.4f}")
    print("="*40 + "\n")

    return best_overall_N, best_overall_model_name, best_overall_clf

def plot_results(error_rates, n_list, model_names, results_dir):
    """Generates and saves visualizations for error rates."""
    print("Generating visualizations...")
    
    # Heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(error_rates, aspect='auto', cmap='viridis')
    plt.colorbar(label='Error Rate')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.yticks(range(len(n_list)), n_list)
    plt.title("Error rate heat map")
    plt.ylabel("N Genes")
    plt.xlabel("Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_rate_heatmap.png"))
    plt.close()

    # Line Plot
    colors = ['indigo', 'magenta', 'blue', 'green', 'red', 'orange', 'purple', 'cyan']
    plt.figure(figsize=(10, 6))
    for i in range(error_rates.shape[1]):
        plt.plot(error_rates[:, i], c=colors[i % len(colors)], label=model_names[i], marker='o')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(len(n_list)), n_list)
    plt.title("Error rate against N-Genes")
    plt.xlabel("N Genes (Index)")
    plt.ylabel("Error Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_rate_gene_subsets.png"))
    plt.close()
    print(f"Plots saved to {results_dir}")

def predict_test_data(test_tdf, best_N, best_clf, le, output_file="output.txt", results_dir="results"):
    """Predicts the classes for the test dataset using the best model and saves the output."""
    print("Predicting test data...")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    x_test_full = test_tdf.drop(['SNO', 'rank'], axis=1).to_numpy().T
    x_test_best_N = x_test_full[:, :best_N]

    # best_clf is already a fitted Pipeline (scaler + model) from GridSearchCV on the training set
    predictions = best_clf.predict(x_test_best_N)
    test_class_labels = le.inverse_transform(predictions.astype(int))
    
    print("Test dataset predictions:")
    print(test_class_labels)

    # Save to file
    out_path = os.path.join(results_dir, output_file)
    with open(out_path, "w") as f:
        for label in test_class_labels:
            f.write(label + '\n')
    print(f"Predictions saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Disease Prediction Model Training and Evaluation")
    parser.add_argument("--train_data", type=str, default=r"datasets\pp5i_train.gr.csv", help="Path to training data CSV")
    parser.add_argument("--test_data", type=str, default=r"datasets\pp5i_test.gr.csv", help="Path to test data CSV")
    parser.add_argument("--class_data", type=str, default=r"datasets\pp5i_train_class.txt", help="Path to class labels TXT")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results and plots")
    
    args = parser.parse_args()

    # Create results dir if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # 1. Load Data
    train_df, test_df, class_df = load_data(args.train_data, args.test_data, args.class_data)

    # 2. Preprocess Data
    train_tdf, test_tdf, train_class, label_encoder = preprocess_data(train_df, test_df, class_df)

    # 3. Define Gene Subsets to Test
    N_List = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

    # 4. Train, Evaluate Models, and find the best configuration
    best_N, best_model_name, best_clf = evaluate_models_on_subsets(
        train_tdf, train_class, N_List, args.results_dir
    )

    # 5. Predict on Test set using the absolute best model
    predict_test_data(test_tdf, best_N, best_clf, label_encoder, output_file="output.txt", results_dir=args.results_dir)

if __name__ == "__main__":
    main()