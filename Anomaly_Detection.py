import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def load_dataset(dataset_name):
    if dataset_name == "Credit Card Transactions":
        dataset = pd.read_csv("card_transdata.csv")
        df=dataset.rename(columns={'fraud': 'label'})

    elif dataset_name == "IoT Sensor Data":
        dataset = pd.read_csv("Occupancy.csv")
        df= dataset.rename(columns={'Occupancy': 'label'})
    else:
        dataset = pd.read_csv("cybersecurity_intrusion_data.csv")
        df= dataset.rename(columns={'attack_detected': 'label'})
    return df

def data_overview_tab(df):
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", len(df.columns) - (1 if 'label' in df.columns else 0))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        if 'label' in df.columns:
            anomaly_rate = (df['label'].sum() / len(df)) * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        else:
            st.metric("Anomaly Rate", "Unknown")

    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info)

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("Numerical Columns")
        st.dataframe(df[numeric_cols].describe())

    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Columns")
        for col in categorical_cols:
            with st.expander(f"Values in '{col}'"):
                value_counts = df[col].value_counts().head(10)
                st.write(value_counts)

    if st.checkbox("Show Feature Distributions"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'label' in numeric_cols:
            numeric_cols = numeric_cols.drop('label')

        cols_to_show = numeric_cols[:6]
        st.write(f"Showing distributions for first {len(cols_to_show)} numerical columns")

        for col in cols_to_show:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            if 'label' in df.columns:
                fig = px.histogram(df, x=col, color='label', title=f"Distribution of {col} by Label")
            st.plotly_chart(fig)

def preprocess_data(df):

    with st.expander("Data Preprocessing Steps", expanded=False):
        st.write("**Step 1:** Checking for missing values:")
        missing_values = df.isnull().sum().sum()
        st.write(f"Missing values found: {missing_values}")

        df_processed = df.copy()

        if missing_values > 0:
            st.write("**Step 2:** Handling missing values:")
            for col in df_processed.columns:
                if df_processed[col].dtype in ['object', 'category']:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)

        st.write("**Step 3:** Handling categorical variables:")
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and 'label' not in categorical_cols:
            st.write(f"Converting categorical columns: {list(categorical_cols)}")
            le = LabelEncoder()
            for col in categorical_cols:
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))

        st.write("**Step 4:** Separating features and labels:")
        if 'label' in df_processed.columns:
            X = df_processed.drop('label', axis=1)
            y = df_processed['label']
            st.write(f"Features shape: {X.shape}")
            st.write(f"Labels shape: {y.shape}")

            if y is not None:
                label_dist = y.value_counts()
                st.write("Label distribution:")
                st.write(label_dist)
        else:
            X = df_processed
            y = None
            st.write(f"No labels found. Features shape: {X.shape}")
            st.info("Running in unsupervised mode")

        st.write("**Step 5:** Scaling numerical features:")
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        st.write("Data preprocessing completed!")
        return X, X_scaled.values, y

def display_results(predictions, true_labels, scores, algorithm_name, X):

    st.subheader(f"{algorithm_name} Results")

    if true_labels is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(true_labels, predictions):.3f}")
        with col2:
            st.metric("Precision", f"{precision_score(true_labels, predictions, zero_division=0):.3f}")
        with col3:
            st.metric("Recall", f"{recall_score(true_labels, predictions, zero_division=0):.3f}")
        with col4:
            st.metric("F1-Score", f"{f1_score(true_labels, predictions, zero_division=0):.3f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(true_labels, predictions)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                           labels=dict(x="Predicted", y="Actual"),
                           title="Confusion Matrix",
                           color_continuous_scale="Blues")
        st.plotly_chart(fig_cm)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anomaly Distribution")
        anomaly_counts = pd.Series(predictions).value_counts()
        labels_map = {0: "Normal", 1: "Anomaly"}
        names = [labels_map.get(k, k) for k in anomaly_counts.index]
        fig_pie = px.pie(values=anomaly_counts.values, names=names,
                         title="Anomaly vs Normal Distribution")
        st.plotly_chart(fig_pie)

    with col2:
        st.subheader("Anomaly Scores")
        fig_hist = go.Figure()
        if true_labels is not None:
            fig_hist.add_trace(go.Histogram(x=scores[true_labels == 0], name="Normal", opacity=0.7))
            fig_hist.add_trace(go.Histogram(x=scores[true_labels == 1], name="Anomaly", opacity=0.7))
            fig_hist.update_layout(title="Score Distribution by True Label", barmode='overlay')
        else:
            fig_hist.add_trace(go.Histogram(x=scores, name="All Points", opacity=0.7))
            fig_hist.update_layout(title="Anomaly Score Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

def isolation_forest(X, X_scaled, y, n_estimators,contamination):
    with st.spinner("Running Isolation Forest:"):
        st.write("**Processing Steps:**")
        st.write("1_Initializing Isolation Forest model")
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

        st.write("2_Training model on scaled data")
        predictions = model.fit_predict(X_scaled)

        st.write("3_Calculating anomaly scores")
        scores = model.score_samples(X_scaled)

        st.write("4_Converting predictions to binary format")
        predictions_binary = np.where(predictions == -1, 1, 0)

        st.success("Isolation Forest completed!")

        display_results(predictions_binary, y, scores, "Isolation Forest", X)
        return predictions_binary, scores

def local_outlier_factor(X, X_scaled, y, n_neighbors,contamination):

    with st.spinner("Running Local Outlier Factor:"):
        st.write("**Processing Steps:**")
        st.write("1_Initializing LOF model")
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )

        st.write("2_Fitting model and predicting anomalies")
        predictions = model.fit_predict(X_scaled)

        st.write("3_Extracting negative outlier factors")
        scores = model.negative_outlier_factor_

        st.write("4_Converting predictions to binary format")
        predictions_binary = np.where(predictions == -1, 1, 0)

        st.success("Local Outlier Factor completed!")

        display_results(predictions_binary, y, scores, "Local Outlier Factor", X)
        return predictions_binary, scores

def one_class_svm(X, X_scaled, y, nu, kernel, gamma):

    with st.spinner("Running One-Class SVM:"):
        st.write("**Processing Steps:**")
        st.write("1_Initializing One-Class SVM model")
        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )

        st.write("2_Training model to learn normal data boundary")
        predictions = model.fit_predict(X_scaled)

        st.write("3_Calculating decision function scores")
        scores = model.score_samples(X_scaled)

        st.write("4_Converting predictions to binary format")
        predictions_binary = np.where(predictions == -1, 1, 0)

        st.success("One-Class SVM completed!")

        display_results(predictions_binary, y, scores, "One-Class SVM", X)
        return predictions_binary, scores


st.set_page_config(
    page_title="Anomaly Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Anomaly Detection Application")
st.markdown("**Detect anomalies in your data using advanced machine learning algorithms**")

st.sidebar.header("Configuration")
dataset_options = ['Credit Card Transactions', 'IoT Sensor Data', 'Network Logs']
dataset_name = st.sidebar.selectbox("Select Dataset", dataset_options)
df = load_dataset(dataset_name)

algorithm_options = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']
algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)
st.sidebar.subheader("Algorithm Parameters")
if algorithm == 'Isolation Forest':
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 300, 100)
    contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.5, 0.1)

elif algorithm == 'Local Outlier Factor':
    n_neighbors = st.sidebar.slider("Number of Neighbors", 5, 50, 20)
    contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.5, 0.1)

else:
    nu = st.sidebar.slider("Nu (outlier fraction)", 0.01, 0.5, 0.1)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    gamma = st.sidebar.selectbox("Gamma", ['scale', 'auto'])

tab1, tab2, tab3 = st.tabs(["Data Overview", "Anomaly Detection", "Exploratory Analysis"])
with tab1:
    data_overview_tab(df)
with tab2:
    st.header(f"{algorithm} Analysis")

    if st.button(f" Run {algorithm}"):
        X, X_scaled, y = preprocess_data(df)
        if algorithm == 'Isolation Forest':
            predictions, scores = isolation_forest(X, X_scaled, y, n_estimators,contamination)
        elif algorithm == 'Local Outlier Factor':
            predictions, scores = local_outlier_factor(X, X_scaled, y, n_neighbors,contamination)
        else:
            predictions, scores = one_class_svm(X, X_scaled, y, nu, kernel, gamma)

        st.subheader("Analysis Summary")
        total_anomalies = np.sum(predictions)
        st.metric("Total Anomalies Detected", int(total_anomalies))

with tab3:
            st.header("Exploratory Data Analysis")

            if st.checkbox("Show Correlation Matrix"):
                numeric_df = df.select_dtypes(include=[np.number])
                if 'label' in numeric_df.columns:
                    numeric_df = numeric_df.drop('label', axis=1)

                if len(numeric_df.columns) > 0:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, title="Feature Correlation Matrix",
                                    color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical columns available for correlation analysis")

            if st.checkbox("Show Box Plots"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if 'label' in numeric_cols:
                    numeric_cols = numeric_cols.drop('label')

                if len(numeric_cols) > 0:
                    cols_to_plot = numeric_cols[:6]
                    st.write(f"Showing box plots for first {len(cols_to_plot)} numerical columns")

                    for col in cols_to_plot:
                        fig = px.box(df, y=col, title=f"Box Plot - {col}")
                        if 'label' in df.columns:
                            fig = px.box(df, y=col, color='label', title=f"Box Plot - {col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical columns available for box plot analysis")
