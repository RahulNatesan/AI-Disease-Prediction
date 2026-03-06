import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger("sklearn").setLevel(logging.ERROR)

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
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction Model",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.55);
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        margin-top: 0.3rem;
    }

    /* Section header */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c4b5fd;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(196,181,253,0.3);
        margin-bottom: 1rem;
    }

    /* Hero */
    .hero-box {
        background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(59,130,246,0.3));
        border: 1px solid rgba(139,92,246,0.4);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-box h1 { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0.3rem 0; }
    .hero-box p  { color: rgba(255,255,255,0.65); font-size: 1rem; margin: 0; }

    /* Prediction badges */
    .pred-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .EPD { background: rgba(239,68,68,0.25);  color: #fca5a5; border: 1px solid #ef4444; }
    .JPA { background: rgba(245,158,11,0.25); color: #fcd34d; border: 1px solid #f59e0b; }
    .MED { background: rgba(59,130,246,0.25); color: #93c5fd; border: 1px solid #3b82f6; }
    .MGL { background: rgba(16,185,129,0.25); color: #6ee7b7; border: 1px solid #10b981; }
    .RHB { background: rgba(139,92,246,0.25); color: #c4b5fd; border: 1px solid #8b5cf6; }

    /* Info tag */
    .info-tag {
        background: rgba(59,130,246,0.2);
        border: 1px solid rgba(59,130,246,0.4);
        color: #93c5fd;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }

    /* Streamlit default text overrides */
    .stMarkdown p, label, .stRadio label { color: rgba(255,255,255,0.85) !important; }
    h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Disease Predictor")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool uses **gene expression microarray data** to classify brain tumours "
        "into one of five disease types:\n\n"
        "- 🔴 **EPD** — Ependymoma\n"
        "- 🟡 **JPA** — Juvenile Pilocytic Astrocytoma\n"
        "- 🔵 **MED** — Medulloblastoma\n"
        "- 🟢 **MGL** — Malignant Glioma\n"
        "- 🟣 **RHB** — Rhabdoid Tumour"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    n_list_options = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
    selected_n_list = st.multiselect(
        "Top-N gene subsets to test",
        options=n_list_options,
        default=[10, 15, 20, 25, 30],
        help="The model will evaluate each selected N and pick the best one."
    )
    use_grid_search = st.toggle("Enable GridSearchCV tuning", value=True,
                                help="Finds the best hyperparameters using 5-fold CV. Slower but more accurate.")
    cv_folds = st.slider("Cross-validation folds", min_value=3, max_value=10, value=5)
    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.4)'>Built with ❤️ using Streamlit & scikit-learn</small>",
        unsafe_allow_html=True
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div style="font-size:3rem">🧬</div>
    <h1>Disease Prediction Model</h1>
    <p>Gene expression microarray classification using machine learning</p>
</div>
""", unsafe_allow_html=True)

# ─── File Upload Section ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Upload Test Data</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="info-tag">ℹ️ Training data and class labels are loaded automatically from the bundled dataset. '
    'Upload your own <b>test CSV</b> below, or leave blank to use the bundled sample test data.</div>',
    unsafe_allow_html=True
)

test_file = st.file_uploader("Test Data (.csv)", type=["csv"], key="test",
                              help="Upload your gene-expression test CSV. Leave blank to use the bundled pp5i test set.")

st.markdown("---")


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_bundled_train():
    """Always load training data and class labels from bundled files."""
    train_df = pd.read_csv("datasets/pp5i_train.gr.csv")
    class_df = pd.read_csv("datasets/pp5i_train_class.txt")
    return train_df, class_df


@st.cache_data(show_spinner=False)
def load_bundled_test():
    """Load the bundled sample test data."""
    return pd.read_csv("datasets/pp5i_test.gr.csv")


def preprocess(train_df, test_df, class_df):
    class_np   = class_df.to_numpy()
    le         = preprocessing.LabelEncoder()
    train_class = le.fit_transform(class_np.ravel())

    ttdf_sno   = train_df['SNO']
    ttdf_rem   = train_df.iloc[:, 1:].clip(20, 16000)
    tsdf_sno   = test_df['SNO']
    tsdf_rem   = test_df.iloc[:, 1:].clip(20, 16000)

    ttdf_cal   = abs(ttdf_rem.max(axis=1) / ttdf_rem.min(axis=1))
    del_ind    = ttdf_cal[ttdf_cal < 2].index

    train_tdf  = pd.concat([ttdf_sno.drop(del_ind), ttdf_rem.drop(del_ind)], axis=1, sort=False)
    test_tdf   = pd.concat([tsdf_sno.drop(del_ind), tsdf_rem.drop(del_ind)], axis=1, sort=False)

    f_vals, _  = f_classif(train_tdf.drop('SNO', axis=1).T, train_class)
    train_tdf['rank'] = f_vals
    test_tdf['rank']  = f_vals
    train_tdf = train_tdf.sort_values('rank', ascending=False)
    test_tdf  = test_tdf.sort_values('rank', ascending=False)
    return train_tdf, test_tdf, train_class, le


def get_models(use_gs):
    models = {
        'GaussianNB':            (GaussianNB(),                              {}),
        'DecisionTree':          (DecisionTreeClassifier(random_state=42),  {'classifier__max_depth': [None, 10], 'classifier__min_samples_split': [2, 5]}),
        'KNN':                   (KNeighborsClassifier(),                   {'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform','distance']}),
        'MLP':                   (MLPClassifier(random_state=42, max_iter=300), {'classifier__hidden_layer_sizes': [(25,25),(50,50)], 'classifier__activation': ['relu','tanh'], 'classifier__solver': ['sgd','adam']}),
        'ExtraTrees':            (ExtraTreesClassifier(random_state=42),    {'classifier__n_estimators': [100, 350], 'classifier__max_depth': [None, 10]}),
        'RandomForest':          (RandomForestClassifier(random_state=42),  {'classifier__n_estimators': [100, 300], 'classifier__max_depth': [None, 10]}),
    }
    if not use_gs:
        # Return empty param grids to skip GridSearchCV
        return {k: (v[0], {}) for k, v in models.items()}
    return models


def run_evaluation(train_tdf, train_class, n_list, cv_folds, use_gs, progress_bar, status_text):
    models_dict  = get_models(use_gs)
    model_names  = list(models_dict.keys())
    error_rates  = np.zeros((len(n_list), len(model_names)))
    results_rows = []

    full_x_train = train_tdf.drop(['SNO', 'rank'], axis=1).to_numpy().T
    best_score, best_N, best_model_name, best_clf = 0, 0, "", None

    total_steps = len(n_list) * len(model_names)
    step = 0

    scoring = {
        'accuracy':  make_scorer(accuracy_score),
        'f1':        make_scorer(f1_score,        average='weighted', zero_division=0),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
    }

    for i, N in enumerate(n_list):
        x_trainN = full_x_train[:, :N]
        for j, model_name in enumerate(model_names):
            status_text.markdown(
                f'<div class="info-tag">⚙️ Training <b>{model_name}</b> with top <b>{N}</b> genes…</div>',
                unsafe_allow_html=True
            )
            base_model, param_grid = models_dict[model_name]
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', base_model)])
            cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            if param_grid and use_gs:
                gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
                gs.fit(x_trainN, train_class)
                best_pipeline = gs.best_estimator_
                best_params   = gs.best_params_
            else:
                pipeline.fit(x_trainN, train_class)
                best_pipeline = pipeline
                best_params   = {}

            scores = cross_validate(best_pipeline, x_trainN, train_class, cv=cv, scoring=scoring)
            acc    = np.mean(scores['test_accuracy'])
            f1     = np.mean(scores['test_f1'])
            prec   = np.mean(scores['test_precision'])
            rec    = np.mean(scores['test_recall'])

            error_rates[i, j] = 1 - acc
            results_rows.append({
                'N Genes': N, 'Model': model_name,
                'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec,
                'Best Params': str(best_params)
            })

            if acc > best_score:
                best_score, best_N, best_model_name, best_clf = acc, N, model_name, best_pipeline

            step += 1
            progress_bar.progress(step / total_steps)

    results_df = pd.DataFrame(results_rows)
    return error_rates, model_names, results_df, best_N, best_model_name, best_clf, best_score


# ─── Run Button ───────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("🚀 Run Analysis", width='stretch')

# ─── Analysis ─────────────────────────────────────────────────────────────────
if run_clicked:
    if not selected_n_list:
        st.warning("Please select at least one N value in the sidebar.")
        st.stop()

    # Load data
    with st.spinner("Loading data…"):
        try:
            train_df, class_df = load_bundled_train()
        except FileNotFoundError:
            st.error("❌ Bundled training data not found. Please ensure 'datasets/pp5i_train.gr.csv' and 'datasets/pp5i_train_class.txt' are present.")
            st.stop()

        if test_file is not None:
            test_df = pd.read_csv(test_file)
        else:
            try:
                test_df = load_bundled_test()
            except FileNotFoundError:
                st.error("❌ No test file uploaded and bundled test data not found. Please upload a test CSV.")
                st.stop()

    # Preprocess
    with st.spinner("Preprocessing & ranking genes…"):
        train_tdf, test_tdf, train_class, le = preprocess(train_df, test_df, class_df)

    st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    classes, counts = np.unique(train_class, return_counts=True)
    class_labels    = le.inverse_transform(classes)
    for col, lbl, cnt in zip([m1,m2,m3,m4][:len(class_labels)], class_labels, counts):
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{cnt}</div>
                <div class="metric-sub">training samples</div>
            </div>''', unsafe_allow_html=True)
    if len(class_labels) > 4:
        col_extra = st.columns(len(class_labels) - 4)
        for col, lbl, cnt in zip(col_extra, class_labels[4:], counts[4:]):
            with col:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value">{cnt}</div>
                    <div class="metric-sub">training samples</div>
                </div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Training Models</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text  = st.empty()

    error_rates, model_names, results_df, best_N, best_model_name, best_clf, best_score = run_evaluation(
        train_tdf, train_class, sorted(selected_n_list), cv_folds, use_grid_search, progress_bar, status_text
    )
    status_text.empty()
    progress_bar.progress(1.0)

    # ── Best Model Banner ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🏆 Best Configuration</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value" style="font-size:1.4rem">{best_model_name}</div>
        </div>''', unsafe_allow_html=True)
    with b2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Best N Genes</div>
            <div class="metric-value">{best_N}</div>
        </div>''', unsafe_allow_html=True)
    with b3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{best_score:.1%}</div>
        </div>''', unsafe_allow_html=True)
    with b4:
        best_row = results_df[(results_df['Model']==best_model_name) & (results_df['N Genes']==best_N)]
        best_f1  = best_row['F1'].values[0] if len(best_row) else 0
        st.markdown(f'''<div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{best_f1:.1%}</div>
        </div>''', unsafe_allow_html=True)

    # ── Results Table ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Full Evaluation Results</div>', unsafe_allow_html=True)
    display_df = results_df[['N Genes','Model','Accuracy','F1','Precision','Recall']].copy()
    display_df[['Accuracy','F1','Precision','Recall']] = display_df[['Accuracy','F1','Precision','Recall']].applymap(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)  # noqa: deprecated but dataframe uses different API

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Visualizations</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    palette = ['#8b5cf6','#ec4899','#3b82f6','#10b981','#f59e0b','#ef4444']
    n_list_sorted = sorted(selected_n_list)

    with chart_col1:
        st.markdown("**Error Rate Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        im = ax.imshow(error_rates, aspect='auto', cmap='plasma')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=40, ha='right', fontsize=8, color='white')
        ax.set_yticks(range(len(n_list_sorted)))
        ax.set_yticklabels(n_list_sorted, color='white')
        ax.set_title("Error Rate (lower=better)", color='white', fontsize=10)
        ax.set_xlabel("Classifier", color='white', fontsize=8)
        ax.set_ylabel("N Genes", color='white', fontsize=8)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with chart_col2:
        st.markdown("**Error Rate vs N Genes**")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        for i, name in enumerate(model_names):
            ax.plot(range(len(n_list_sorted)), error_rates[:, i],
                    marker='o', color=palette[i % len(palette)], label=name, linewidth=2)
        ax.set_xticks(range(len(n_list_sorted)))
        ax.set_xticklabels(n_list_sorted, color='white')
        ax.tick_params(colors='white')
        ax.set_title("Error Rate vs N Genes", color='white', fontsize=10)
        ax.set_xlabel("N Genes", color='white', fontsize=8)
        ax.set_ylabel("Error Rate", color='white', fontsize=8)
        legend = ax.legend(fontsize=7, facecolor='#2e2b6e', edgecolor='none', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.2))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Accuracy bar chart per model ───────────────────────────────────────────
    st.markdown("**Accuracy per Model (Best N for each)**")
    best_per_model = results_df.loc[results_df.groupby('Model')['Accuracy'].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#1e1b4b')
    ax.set_facecolor('#1e1b4b')
    bars = ax.barh(best_per_model['Model'], best_per_model['Accuracy'],
                   color=palette[:len(best_per_model)], edgecolor='none', height=0.5)
    for bar, val in zip(bars, best_per_model['Accuracy']):
        ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va='center', ha='right', color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy", color='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.set_title("Best Accuracy per Classifier", color='white', fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.2))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Test Predictions ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Test Dataset Predictions</div>', unsafe_allow_html=True)

    x_test_full  = test_tdf.drop(['SNO','rank'], axis=1).to_numpy().T
    x_test_bestN = x_test_full[:, :best_N]
    preds        = best_clf.predict(x_test_bestN)
    labels       = le.inverse_transform(preds.astype(int))

    st.markdown("Predicted disease class for each test patient:")
    badge_html = ""
    for i, label in enumerate(labels):
        badge_html += f'<span class="pred-badge {label}">P{i+1}: {label}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    pred_counts = pd.Series(labels).value_counts().reset_index()
    pred_counts.columns = ['Disease', 'Count']
    st.markdown("<br>", unsafe_allow_html=True)
    pc1, pc2 = st.columns([1, 2])
    with pc1:
        st.table(pred_counts)
    with pc2:
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        colors_pie = ['#ef4444','#f59e0b','#3b82f6','#10b981','#8b5cf6']
        wedges, texts, autotexts = ax.pie(
            pred_counts['Count'],
            labels=pred_counts['Disease'],
            autopct='%1.1f%%',
            colors=colors_pie[:len(pred_counts)],
            startangle=90,
            textprops={'color':'white', 'fontsize':9}
        )
        ax.set_title("Prediction Distribution", color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Download predictions
    pred_df      = pd.DataFrame({'Patient': [f'P{i+1}' for i in range(len(labels))], 'Predicted Disease': labels})
    csv_bytes    = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions (CSV)", data=csv_bytes,
                       file_name="predictions.csv", mime="text/csv")

elif not run_clicked:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); 
                border-radius:16px; padding:2rem; text-align:center; color:rgba(255,255,255,0.5)">
        <div style="font-size:3rem">🧬</div>
        <h3 style="color:rgba(255,255,255,0.6)!important">Ready to Analyse</h3>
        <p>Configure your settings in the sidebar, then click <b>Run Analysis</b> to begin.</p>
    </div>
    """, unsafe_allow_html=True)
