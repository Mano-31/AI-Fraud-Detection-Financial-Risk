import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/fraud_detection_pipeline.pkl"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Fraud Detection System")

st.sidebar.info(
    """
    AI-Powered Fraud Detection & 
    Financial Risk Intelligence System

    Model:
    XGBoost

    Techniques:
    • Feature Engineering
    • StandardScaler
    • SMOTE
    • Hyperparameter Tuning
    """
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("💳 AI-Powered Fraud Detection System")

st.write(
    "Upload a transaction dataset to detect fraudulent "
    "transactions using an XGBoost machine learning pipeline."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload Transaction CSV",
    type=["csv"]
)


# =========================================================
# DATASET PROCESSING
# =========================================================

if uploaded_file is not None:

    try:

        # =================================================
        # READ DATA
        # =================================================

        data = pd.read_csv(uploaded_file)

        st.success(
            "✅ Dataset uploaded successfully!"
        )


        # =================================================
        # REQUIRED COLUMNS
        # =================================================

        required_columns = [
            "Time",
            "Amount"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:

            st.error(
                f"❌ Missing required columns: "
                f"{missing_columns}"
            )

            st.stop()


        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        data["Amount_log"] = np.log1p(
            data["Amount"]
        )

        data["Time_hours"] = (
            data["Time"] / 3600
        )


        # =================================================
        # CREATE FEATURES
        # =================================================

        if "Class" in data.columns:

            X = data.drop(
                "Class",
                axis=1
            )

        else:

            X = data.copy()


        # =================================================
        # PREDICTION
        # =================================================

        predictions = model.predict(X)

        probabilities = model.predict_proba(
            X
        )[:, 1]


        # =================================================
        # ADD PREDICTION RESULTS
        # =================================================

        data["Prediction"] = predictions

        data["Fraud_Probability"] = probabilities

        data["Result"] = np.where(
            predictions == 1,
            "Fraud",
            "Genuine"
        )


        # =================================================
        # RISK LEVEL
        # =================================================

        data["Risk_Level"] = np.where(
            probabilities >= 0.80,
            "🔴 High Risk",
            np.where(
                probabilities >= 0.50,
                "🟠 Medium Risk",
                "🟢 Low Risk"
            )
        )


        # =================================================
        # CALCULATE DASHBOARD VALUES
        # =================================================

        total_transactions = len(data)

        fraud_count = int(
            (predictions == 1).sum()
        )

        genuine_count = int(
            (predictions == 0).sum()
        )

        fraud_percentage = (
            fraud_count /
            total_transactions *
            100
        )

        high_risk_count = int(
            (
                data["Fraud_Probability"]
                >= 0.80
            ).sum()
        )


        # =================================================
        # PROFESSIONAL KPI DASHBOARD
        # =================================================

        st.divider()

        st.subheader(
            "📊 Fraud Detection Dashboard"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "💳 Total Transactions",
                f"{total_transactions:,}"
            )


        with col2:

            st.metric(
                "🟢 Genuine Transactions",
                f"{genuine_count:,}"
            )


        with col3:

            st.metric(
                "🚨 Fraud Detected",
                f"{fraud_count:,}"
            )


        with col4:

            st.metric(
                "🔴 High Risk",
                f"{high_risk_count:,}"
            )


        st.info(
            f"Fraud Detection Rate: "
            f"**{fraud_percentage:.2f}%**"
        )


        # =================================================
        # FRAUD VS GENUINE CHART
        # =================================================

        st.subheader(
            "📈 Transaction Classification"
        )

        classification_data = pd.DataFrame(
            {
                "Transaction Type": [
                    "Genuine",
                    "Fraud"
                ],
                "Count": [
                    genuine_count,
                    fraud_count
                ]
            }
        )

        st.bar_chart(
            classification_data.set_index(
                "Transaction Type"
            )
        )


        # =================================================
        # RISK DISTRIBUTION
        # =================================================

        st.subheader(
            "⚠️ Risk Distribution"
        )

        risk_counts = (
            data["Risk_Level"]
            .value_counts()
        )

        st.bar_chart(
            risk_counts
        )


        # =================================================
        # ROC-AUC CURVE
        # =================================================

        if "Class" in data.columns:

            st.divider()

            st.subheader(
                "📈 ROC-AUC Curve"
            )

            actual = data["Class"]

            auc_score = roc_auc_score(
                actual,
                probabilities
            )

            fpr, tpr, thresholds = roc_curve(
                actual,
                probabilities
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            ax.plot(
                fpr,
                tpr,
                label=(
                    f"ROC-AUC = "
                    f"{auc_score:.4f}"
                )
            )


            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )


            ax.set_xlabel(
                "False Positive Rate"
            )

            ax.set_ylabel(
                "True Positive Rate"
            )

            ax.set_title(
                "ROC-AUC Curve"
            )

            ax.legend()


            st.pyplot(fig)


            # =============================================
            # CONFUSION MATRIX
            # =============================================

            st.subheader(
                "📊 Confusion Matrix"
            )


            cm = confusion_matrix(
                actual,
                predictions
            )


            fig_cm, ax_cm = plt.subplots(
                figsize=(6, 4)
            )


            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[
                    "Genuine",
                    "Fraud"
                ],
                yticklabels=[
                    "Genuine",
                    "Fraud"
                ],
                ax=ax_cm
            )


            ax_cm.set_xlabel(
                "Predicted"
            )

            ax_cm.set_ylabel(
                "Actual"
            )

            ax_cm.set_title(
                "Fraud Detection Confusion Matrix"
            )


            st.pyplot(fig_cm)


            # =============================================
            # CONFUSION MATRIX VALUES
            # =============================================

            tn, fp, fn, tp = cm.ravel()


            st.write(
                "### 🔎 Confusion Matrix Details"
            )


            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.metric(
                    "True Negative",
                    tn
                )


            with c2:

                st.metric(
                    "False Positive",
                    fp
                )


            with c3:

                st.metric(
                    "False Negative",
                    fn
                )


            with c4:

                st.metric(
                    "True Positive",
                    tp
                )


        # =================================================
        # PREDICTION RESULTS
        # =================================================

        st.divider()

        st.subheader(
            "🔍 Prediction Results"
        )


        display_columns = [
            "Time",
            "Amount",
            "Fraud_Probability",
            "Result",
            "Risk_Level"
        ]


        st.dataframe(
            data[display_columns],
            use_container_width=True
        )


        # =================================================
        # DOWNLOAD RESULTS
        # =================================================

        csv_data = data.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )


        # =================================================
        # FRAUD TRANSACTIONS
        # =================================================

        st.divider()

        st.subheader(
            "🚨 Fraud Transactions"
        )


        fraud_data = data[
            data["Prediction"] == 1
        ]


        if len(fraud_data) > 0:

            st.dataframe(
                fraud_data,
                use_container_width=True
            )

        else:

            st.success(
                "✅ No fraud transactions detected."
            )


        # =================================================
        # HIGH-RISK TRANSACTIONS
        # =================================================

        st.subheader(
            "🔴 High Risk Transactions"
        )


        high_risk_data = data[
            data["Fraud_Probability"] >= 0.80
        ]


        if len(high_risk_data) > 0:

            st.dataframe(
                high_risk_data,
                use_container_width=True
            )

        else:

            st.success(
                "✅ No high-risk transactions detected."
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            f"❌ Error processing dataset: {e}"
        )


# =========================================================
# NO FILE UPLOADED
# =========================================================

else:

    st.info(
        "👆 Please upload a transaction CSV file "
        "to start fraud detection."
    )


# =========================================================
# SINGLE TRANSACTION PREDICTION
# =========================================================

st.divider()

st.header(
    "🔎 Single Transaction Fraud Prediction"
)

st.write(
    "Enter transaction details manually to predict "
    "whether the transaction is Genuine or Fraud."
)


with st.form(
    "single_transaction_form"
):

    time_value = st.number_input(
        "Time",
        min_value=0.0,
        value=10000.0
    )


    amount_value = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )


    st.write(
        "### V1 - V28 Features"
    )


    v_values = []


    cols = st.columns(4)


    for i in range(1, 29):

        with cols[(i - 1) % 4]:

            value = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"v{i}"
            )

            v_values.append(value)


    submit = st.form_submit_button(
        "🔍 Predict Transaction"
    )


# =========================================================
# SINGLE TRANSACTION RESULT
# =========================================================

if submit:

    # -----------------------------------------------------
    # CREATE TRANSACTION
    # -----------------------------------------------------

    transaction = {
        "Time": time_value
    }


    for i in range(28):

        transaction[
            f"V{i + 1}"
        ] = v_values[i]


    transaction["Amount"] = amount_value


    transaction["Amount_log"] = np.log1p(
        amount_value
    )


    transaction["Time_hours"] = (
        time_value / 3600
    )


    transaction_data = pd.DataFrame(
        [transaction]
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        transaction_data
    )[0]


    probability = model.predict_proba(
        transaction_data
    )[0][1]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if prediction == 1:

        st.error(
            "🚨 FRAUD TRANSACTION DETECTED"
        )

    else:

        st.success(
            "✅ GENUINE TRANSACTION"
        )


    # -----------------------------------------------------
    # FRAUD PROBABILITY
    # -----------------------------------------------------

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )


    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    if probability >= 0.80:

        st.error(
            "🔴 High Risk"
        )

    elif probability >= 0.50:

        st.warning(
            "🟠 Medium Risk"
        )

    else:

        st.success(
            "🟢 Low Risk"
        )