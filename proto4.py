import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Streamlit app
def automl_app():
    st.title("AutoML Application")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Data preprocessing
        st.sidebar.header("Preprocessing Options")
        missing_strategy = st.sidebar.selectbox(
            "Select Missing Value Strategy",
            ["Automatic", "Mean", "Median", "Mode"]
        )

        # Handle missing values
        if missing_strategy == "Automatic":
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
            for col in data.select_dtypes(include=[object]).columns:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].mode()[0], inplace=True)
        elif missing_strategy == "Mean":
            for col in data.select_dtypes(include=[np.number]).columns:
                data[col].fillna(data[col].mean(), inplace=True)
        elif missing_strategy == "Median":
            for col in data.select_dtypes(include=[np.number]).columns:
                data[col].fillna(data[col].median(), inplace=True)
        elif missing_strategy == "Mode":
            for col in data.select_dtypes(include=[object]).columns:
                data[col].fillna(data[col].mode()[0], inplace=True)

        # Feature and target selection
        st.sidebar.header("Feature Selection")
        target_column = st.sidebar.selectbox("Select Target Column", data.columns)

        # Automatically select all columns except the target column as features by default
        default_features = [col for col in data.columns if col != target_column]
        feature_columns = st.sidebar.multiselect(
            "Select Feature Columns",
            options=[col for col in data.columns if col != target_column],
            default=default_features  # Preselect all feature columns by default
        )

        if not feature_columns:
            st.error("Please select at least one feature column.")
            return

        # Encoding Options
        encoding_method = st.sidebar.selectbox(
            "Select Encoding Method",
            ["Automatic (Best)", "Label Encoding", "One-Hot Encoding"]
        )

        # Automatic encoding method selection
        if encoding_method == "Automatic (Best)":
            categorical_columns = data.select_dtypes(include=[object]).columns
            if len(categorical_columns) > 0:
                # Prefer One-Hot Encoding if there are multiple categories
                encoding_method = "One-Hot Encoding"
            else:
                encoding_method = "Label Encoding"
        
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            for col in data.select_dtypes(include=[object]).columns:
                data[col] = le.fit_transform(data[col])

        elif encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data, drop_first=True)

        X = data[feature_columns]
        y = data[target_column]

        # Train-test split
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Model selection
        st.sidebar.header("Model Selection")
        task = st.sidebar.radio("Task Type", ["Regression", "Classification"])

        # Algorithms and metrics
        model_metrics = []
        if task == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
            }
        else:  # Classification
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVM": SVC(probability=True),
                "Naive Bayes": GaussianNB(),
            }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                model_metrics.append({"Model": name, "MSE": mse, "R2": r2})
            else:  # Classification
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")
                model_metrics.append(
                    {"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
                )

        # Display metrics
        metrics_df = pd.DataFrame(model_metrics)
        st.subheader("Model Metrics")
        st.dataframe(metrics_df)

        # Determine the best model
        if task == "Regression":
            best_model_name = metrics_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
        else:
            best_model_name = metrics_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]

        st.subheader(f"Best Model: {best_model_name}")

        best_model = models[best_model_name]
        y_pred_best = best_model.predict(X_test)

        # Confusion matrix for classification
        if task == "Classification":
            cm = confusion_matrix(y_test, y_pred_best)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
            disp.plot(ax=ax, cmap="viridis")
            st.pyplot(fig)

        # Graphical results
        st.subheader("Graphical Results")
        fig, ax = plt.subplots()
        if task == "Regression":
            ax.scatter(y_test, y_pred_best, alpha=0.5)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Regression Results")
        else:  # Classification
            probas = best_model.predict_proba(X_test)
            ax.bar(range(len(probas)), probas[:, 1], alpha=0.5)
            ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

        # Download best model
        import pickle

        model_filename = f"{best_model_name.replace(' ', '_')}_model.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(best_model, f)
        st.download_button(
            label="Download Best Model",
            data=open(model_filename, "rb"),
            file_name=model_filename,
        )


if __name__ == "__main__":
    automl_app()
