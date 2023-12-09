import streamlit as st
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components
from numpy import ndarray
from typing import Any
from tqdm import tqdm
import joblib
from sklearn import metrics
import matplotlib
import os
matplotlib.rcParams['animation.embed_limit'] = 2**128



def load_data() -> DataFrame | None:
    st.markdown("### 1. Data Loading")
    data_source_options = ["Use Sklearn Dataset", "Upload Custom Dataset"]
    sklearn_dataset_options = ["Iris", "Breast Cancer", "Wine"]

    data_source = st.selectbox(label="Select Data Source", options=data_source_options) 
    df = None

    if data_source == data_source_options[0]:
        selected_dataset = st.selectbox(label="Choose Sklearn Dataset", options=sklearn_dataset_options)
        data = None
        if selected_dataset == sklearn_dataset_options[0]:
            data = load_iris()
        elif selected_dataset == sklearn_dataset_options[1]:
            data = load_breast_cancer()
        elif selected_dataset == sklearn_dataset_options[2]:
            data = load_wine()
        
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        df["target"] = data.target
    
    elif data_source == data_source_options[1]:
        file = st.file_uploader("Upload CSV file here", type=[".csv"])
        if file is not None:
            df = pd.DataFrame(file)
    
    return df



def preprocess_data(df: DataFrame) -> DataFrame:
    st.markdown("### 2. Data Preprocessing")
    columns = df.columns
    preprocessing_options = ["Null Rows Dropping", "Standard Scaling", "Min-max Scaling"]
    preprocessing_methods = st.multiselect(label="Select preprocessing methods:", options=preprocessing_options)
    columns_to_apply = st.multiselect(label="Apply to columns:", options=columns)
    if st.button(label="Apply"):
        if preprocessing_options[0] in preprocessing_methods:
            df = df.dropna(axis=0, subset=columns_to_apply)
        if preprocessing_options[1] in preprocessing_methods:
            scaler = StandardScaler()
            df = df.apply(lambda col: scaler.fit_transform([[x] for x in col]).flatten() if col.name in columns_to_apply else col)
        if preprocessing_options[2] in preprocessing_methods:
            scaler = MinMaxScaler()
            df = df.apply(lambda col: scaler.fit_transform([[x] for x in col]).flatten() if col.name in columns_to_apply else col)

        st.dataframe(df)

    return df



def train_test(df) -> None:
    st.markdown("### 2. Training")

    st.markdown("##### 2.1 SVM hyper-parameter configuration")

    C = st.selectbox("C (Regularization Parameter)", [1.000, "custom"])
    if C == "custom": C = st.slider("_", 0.001, 10.000, 1.000, 0.001, format="%.3f", label_visibility="hidden")
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    if kernel == "poly": degree = st.selectbox("Degree (Poly Kernel)", range(1, 11))
    gamma = st.selectbox("Gamma", ["scale", "auto", "custom"])
    if gamma == "custom": gamma = st.slider("_", 0.000, 10.000, 1.000, 0.001, format="%.3f", label_visibility="hidden")
    coef0 = st.selectbox("Coef0", [0.0, "custom"])
    if coef0 == "custom": coef0 = st.slider("_", 0.000, 10.000, 1.000, 0.001, format="%.3f", label_visibility="hidden")
    shrinking = st.selectbox("Shrinking", [True, False])
    probability = st.selectbox("Probability Estimates", [False, True])
    tol = st.selectbox("Tolerance", [1e-3, "custom"])
    if tol == "custom": tol = st.slider("_", 0.000, 10.000, 1.000, 0.001, format="%.3f", label_visibility="hidden")
    cache_size = st.selectbox("Cache Size (MB)", [200, "custom"])
    if cache_size == "custom": cache_size = st.slider("_", 0, 1000, 200, 10, label_visibility="hidden")
    verbose = st.selectbox("Verbose Output", [False, True])
    max_iter = st.selectbox("Max Iterations", [10, "custom"])
    if max_iter == "custom": max_iter = st.slider("_", 10, 1000, 30, 10, label_visibility="hidden")
    decision_function_shape = st.selectbox("Decision Function Shape", ["ovr", "ovo"])
    break_ties = st.selectbox("Break Ties", [False, True])
    random_state = st.number_input("Random State", 42)
    st.markdown("Learn more about SVM parameters in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)")

    svm_model = SVC(
        C=C,
        kernel=kernel,
        degree=degree if kernel == "poly" else 3,  
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=None,  
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state
    )

    st.markdown("##### 2.2 Training Data Preparation")

    features = df.columns
    input_features = st.multiselect(label="Select Input Features:", options=features, max_selections=len(features) - 1)
    target_feature = st.multiselect(label="Select Target Feature:", options=[feature for feature in features if feature not in input_features], max_selections=1)
    X = np.array(df[input_features])
    y = np.array(df[target_feature])
    use_train_test_split = False
    X_train, y_train = X, y.flatten()
    X_test, y_test = None, None
    if st.checkbox("Use train-test splitting"):
        test_size = st.slider("Select Test Size:", 0.1, 0.5, value=0.3,step=0.01)
        splitting_random_state = st.number_input("Train-test split random state:", value=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=splitting_random_state)
        use_train_test_split = True
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    st.markdown("##### 2.3 SVM Training")

    if st.button("Train"):
        if len(input_features) == 0 or len(target_feature) == 0:
            st.warning("Please select input features and target feature.")

        else:
            svm_model.fit(X_train, y_train)
            if len(input_features) == 2:
                train_and_plot_svm_animation(X_train, y_train, input_features, svm_model)      


    st.markdown("### 3. Evaluation")

    if not use_train_test_split:
        test_file = st.file_uploader("Upload CSV file for evaluation here:")
        if test_file is not None:
            test_df = pd.DataFrame(test_file)
            X_test = np.array(test_df[input_features])
            y_test = np.array(test_df[target_feature]).flatten()

    if st.button("Evaluate"):
        if not os.path.isfile("svm_model_checkpoint.joblib"):
            st.warning("You haven't trained the model yet") 
            return
        svm_model.fit(X_train, y_train)
        evaluate_svm_performance(svm_model, X_test, y_test)

    

        
def train_and_plot_svm_animation(X: ndarray, y: ndarray, features, svm_model: SVC) -> SVC:
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolors='k', marker='o')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    Z = []
    V = []
    model_filename = "svm_model_checkpoint.joblib"
    progress_bar = st.progress(0)
    model_params = svm_model.get_params()
    max_iter = model_params["max_iter"]

    for it in tqdm(range(1, max_iter + 1), desc="Training SVM"):
        model_params["max_iter"] = it
        if it > 1:
            svm_model = joblib.load(model_filename)

        svm_model.set_params(**model_params)
        svm_model.fit(X, y)
        joblib.dump(svm_model, model_filename)
        Z.append(svm_model.predict(xy).reshape(xx.shape))
        V.append(svm_model.support_vectors_)
        progress_bar.progress(it / max_iter, text=f"Iteration: {it}")

    def update(frame, Z, V):
        if frame == 0:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolors='k', marker='o')
        else:
            z = Z[frame]
            ax.clear()
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolors='k', marker='o')
            ax.contourf(xx, yy, z, cmap='viridis', alpha=0.3)
            ax.scatter(V[frame][:, 0], V[frame][:, 1], s=100, facecolors='none', edgecolors='k')
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_title(f'SVM Training - Iteration {frame + 1}')

    anim = FuncAnimation(fig, update, fargs=(Z, V), frames=max_iter, repeat=False)
    anim.embed_limit = 100000000
    components.html(anim.to_jshtml(), height=600)
    svm_model = joblib.load(model_filename)
    svm_model.fit(X, y)
    return svm_model


def evaluate_svm_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    classification_rep = metrics.classification_report(y_test, y_pred)
    st.text("Classification Report:")
    st.text(classification_rep)

    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred)

    st.text(f"\nF1-Score: {f1:.2f}")
    st.text(f"Precision: {precision:.2f}")
    st.text(f"Recall: {recall:.2f}")
    st.text(f"Accuracy: {accuracy:.2f}")

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion_matrix, classes=np.unique(y_test), title="Confusion Matrix")



def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap='Blues'):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

        

def main() -> None:
    st.set_page_config(
        page_title="SVM Demo",
    )

    title_html = """<h1 style='text-align: center;'>Support Vector Machine Demo</h1><br><br>"""
    st.markdown(title_html, unsafe_allow_html=True)
    
    df = load_data()
    if df is not None:
        st.dataframe(df, use_container_width=True)
        df = preprocess_data(df)
        train_test(df)
        


if __name__ == "__main__":
    main()