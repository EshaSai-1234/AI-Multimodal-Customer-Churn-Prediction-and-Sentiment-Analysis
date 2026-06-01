# predictive_model.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import shap

class EvoBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Research-driven EvoBoost implementation for tabular classification tasks.
    Uses probabilistic residuals for streamlined boosting adjustments and enhanced generalization.
    """
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42):
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
        
        # 1. Initialize baseline prediction with log-odds probability tracking
        p_mean = np.mean(y_arr)
        self.base_pred_ = np.log(p_mean / (1.0 - p_mean) if p_mean > 0 else 0.01)
        F_t = np.full(shape=y_arr.shape, fill_value=self.base_pred_)
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # 2. EvoBoost Probabilistic Residual Computation: (Actual Target - Predicted Probability)
            p_t = self._sigmoid(F_t)
            residuals = y_arr - p_t
            
            # 3. Fit a weak learner base tree directly onto calculated residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_arr, residuals)
            
            # 4. Scale and update the predictive functional state accumulation
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

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def run_predictive_churn_pipeline(data_path: str):
    # 1. Read input customer dataset
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['customer_id', 'latest_support_log', 'churn'])
    y = df['churn']
    customer_ids = df['customer_id']
    raw_logs = df['latest_support_log']
    
    # 2. Convert categorical profiles safely into numeric space
    X_encoded = pd.get_dummies(X, columns=['contract_type'], drop_first=True, dtype=int)
    
    # 3. Partition into analytical testing profiles without resampling
    X_train, X_test, y_train, y_test, idx_train, idx_test, logs_train, logs_test = train_test_split(
        X_encoded, y, customer_ids, raw_logs,
        test_size=0.25, random_state=42, stratify=y
    )
    
    # 5. Model Initialization and Fitting using the EvoBoost Framework
    model = EvoBoostClassifier(n_estimators=150, learning_rate=0.04, max_depth=4)
    model.fit(X_train, y_train)
    
    # 6. Extract Cohorts flagged above high threat probability levels (>= 0.60)
    test_probs = model.predict_proba(X_test)[:, 1]
    
    evaluation_frame = pd.DataFrame({
        'customer_id': idx_test.values,
        'churn_risk': test_probs,
        'top_negative_reason': logs_test.values,
        'tenure': X_test['tenure_months'].values
    })
    
    actionable_at_risk_cohort = evaluation_frame[evaluation_frame['churn_risk'] >= 0.60]
    
    # 7. Compute SHAP Values for Model Interpretation
    shap_matrix = None
    try:
        explainer = shap.Explainer(model.predict, X_train, feature_names=X_train.columns.tolist())
        shap_matrix = explainer(X_test)
    except Exception as exc:
        print(f"[Model Status] SHAP explanation skipped due to: {exc}")
        shap_matrix = None
    
    print(f"[Model Status] Pipeline evaluation completed successfully.")
    print(f"[Model Status] Identified {len(actionable_at_risk_cohort)} targets requiring retention rescue.")
    return actionable_at_risk_cohort, shap_matrix

if __name__ == "__main__":
    try:
        cohort, shap_vals = run_predictive_churn_pipeline('customer_churn_400.csv')
        print("\n--- Isolated At-Risk Customers (Sample View) ---")
        print(cohort.head())
    except FileNotFoundError:
        print("Dataset missing. Run your data generation script to output 'customer_churn_400.csv' first.")