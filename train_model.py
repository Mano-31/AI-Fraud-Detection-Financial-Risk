import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ============================================
# 1. Load Dataset
# ============================================

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_detection_pipeline.pkl"

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)


# ============================================
# 2. Remove Duplicates
# ============================================

print("\nRemoving duplicate rows...")

data = data.drop_duplicates().copy()

print("Shape after removing duplicates:", data.shape)


# ============================================
# 3. Feature Engineering
# ============================================

print("\nCreating new features...")

data["Amount_log"] = np.log1p(data["Amount"])
data["Time_hours"] = data["Time"] / 3600


# ============================================
# 4. Separate Features and Target
# ============================================

X = data.drop("Class", axis=1)
y = data["Class"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================
# 5. Train-Test Split
# ============================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 6. Create ML Pipeline
# ============================================

print("\nCreating ML pipeline...")

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("smote", SMOTE(random_state=42)),

    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])


# ============================================
# 7. Train Model
# ============================================

print("\nTraining XGBoost model...")

final_pipeline.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# 8. Create Models Folder
# ============================================

os.makedirs("models", exist_ok=True)


# ============================================
# 9. Save Model
# ============================================

print("\nSaving model...")

joblib.dump(
    final_pipeline,
    MODEL_PATH
)

print("Model saved successfully!")
print("Model path:", MODEL_PATH)


# ============================================
# 10. Test Saved Model
# ============================================

print("\nTesting saved model...")

loaded_model = joblib.load(MODEL_PATH)

sample_predictions = loaded_model.predict(X_test.head(5))

print("\nSample predictions:")
print(sample_predictions)

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")