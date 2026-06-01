# evaluation_metrics.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================================================================
# CUSTOM CORE PREDICTIVE ENGINE (EVOBOOST VECTOR TOPOLOGY)
# =====================================================================
class EvoBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=150, learning_rate=0.04, max_depth=4, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.base_pred_ = 0.0

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y, dtype=np.float64)
        np.random.seed(self.random_state)
        
        p_mean = np.mean(y_arr)
        self.base_pred_ = np.log(p_mean / (1.0 - p_mean) if p_mean > 0 else 0.01)
        F_t = np.full(shape=y_arr.shape, fill_value=self.base_pred_)
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            p_t = self._sigmoid(F_t)
            residuals = y_arr - p_t
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_arr, residuals)
            
            F_t += self.learning_rate * tree.predict(X_arr)
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        X_arr = np.array(X)
        F_t = np.full(shape=(X_arr.shape[0],), fill_value=self.base_pred_)
        for tree in self.estimators_:
            F_t += self.learning_rate * tree.predict(X_arr)
        probabilities = self._sigmoid(F_t)
        return np.vstack((1.0 - probabilities, probabilities)).T

    def predict(self, X, threshold=0.60):
        """Generates binary classification predictions based on a specific decision threshold alpha."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# =====================================================================
# EVALUATION PIPELINE RUNNER
# =====================================================================
def main():
    print("=====================================================================")
    print("STARTING MACHINE LEARNING ENGINE ACCURACY & PRECISION EVALUATION")
    print("=====================================================================\n")
    
    data_path = "ai_churn_advanced_nlp_enriched.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"[1/3] Enriched dataset loaded successfully from '{data_path}'.")
    except FileNotFoundError:
        print(f"❌ [Error] File '{data_path}' not found.")
        print("   Please execute 'python nlp_sentiment_enricher.py' first to build the file.")
        return

    # Prepare features and target labels
    X = df.drop(columns=['customer_id', 'latest_support_log', 'churn'])
    y = df['churn']
    
    # One-hot encode string columns safely into pure integer bits
    X_encoded = pd.get_dummies(X, columns=['contract_type', 'nlp_primary_aspect'], drop_first=True, dtype=int)
    
    # Perform a stratified split to keep target distribution balanced across sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"[2/3] Data matrices split successfully. Test Size Matrix: {X_test.shape[0]} profiles.")
    
    # Train the custom EvoBoost Classifier
    print("[3/3] Training model estimators across 150 sequential gradient steps...")
    model = EvoBoostClassifier(n_estimators=150, learning_rate=0.04, max_depth=4)
    model.fit(X_train, y_train)
    
    # Generate test inferences across the operational LangGraph threshold alpha = 0.60
    y_pred = model.predict(X_test, threshold=0.60)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print metrics report
    print("\n" + "="*45)
    print("      EVOBOOST CLASSIFICATION PERFORMANCE METRICS")
    print("="*45)
    print(f"  Accuracy Score  : {accuracy * 100:.2f}%  (Overall Correct Decisions)")
    print(f"  Precision Score : {precision * 100:.2f}%  (Correctness of Churn Flags)")
    print(f"  Recall Score    : {recall * 100:.2f}%  (Coverage of Total True Churners)")
    print(f"  F1-Score Metric : {f1 * 100:.2f}%  (Harmonic Mean Optimization)")
    print("-"*45)
    print("  Confusion Matrix Topology:")
    print(f"    True Negatives  (Retained Correctly) : {cm[0][0]}")
    print(f"    False Positives (False Alarms)        : {cm[0][1]}")
    print(f"    False Negatives (Missed Churners)     : {cm[1][0]}")
    print(f"    True Positives  (Rescued Correctly)  : {cm[1][1]}")
    print("=====================================================================")

if __name__ == "__main__":
    main()