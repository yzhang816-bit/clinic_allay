import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

class RelationalLayer(BaseEstimator):
    """
    Simulates the GNN layer. 
    In this simplified version, it acts as a feature extractor/embedder.
    We use a Random Forest as the backbone for better performance on tabular data.
    """
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        # Return the output of the last hidden layer? 
        # For simplicity, we just return the probability output or raw features 
        # transformed. 
        # Actually, let's just use the MLP as the backbone.
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1) # Return risk score

    def get_embedding(self, X):
        # A hack to get activations would be complex with sklearn MLP.
        # We will just assume the "embedding" is the input features for now 
        # or the high-level risk score.
        return X

class CausalLayer:
    """
    Simulates Causal Inference Engine.
    It identifies 'causal' drivers using feature importance from a linear model
    and filters out noise.
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.causal_features = None
        self.explainer_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

    def fit(self, X, y, feature_names):
        self.explainer_model.fit(X, y)
        coefs = np.abs(self.explainer_model.coef_[0])
        # Select top features
        self.causal_features = [feature_names[i] for i in range(len(coefs)) if coefs[i] > self.threshold]
        if not self.causal_features: # Fallback if regularization is too strong
             top_idx = np.argsort(coefs)[-3:]
             self.causal_features = [feature_names[i] for i in top_idx]
        return self

    def transform(self, X, feature_names):
        # Return only causal features
        df = pd.DataFrame(X, columns=feature_names)
        return df[self.causal_features].values

class SymbolicLayer:
    """
    Simulates Symbolic Reasoning.
    Applies logical rules to adjust predictions.
    Rule: IF Sepsis AND High Lactate THEN High Risk.
    """
    def __init__(self):
        pass

    def apply_rules(self, X, y_prob, feature_names, weight=0.2):
        df = pd.DataFrame(X, columns=feature_names)
        
        # Calculate a "Rule Risk Score"
        rule_score = np.zeros(len(X))
        
        # Factor 1: High Lactate (0-0.8)
        # Increased weight because it's the strongest signal
        lactate_col = 'lab_50813_max'
        if lactate_col in df.columns:
            vals = df[lactate_col].values
            # Sigmoid-like scaling: 1.5 -> 0.1, 4.0 -> 0.8
            rule_score += np.clip((vals - 1.5) / 4.0, 0, 0.8)

        # Factor 2: Sepsis (0.5)
        sepsis_cols = ['diag_A419', 'diag_99591']
        has_sepsis = np.zeros(len(X), dtype=bool)
        for col in sepsis_cols:
            if col in df.columns:
                has_sepsis |= (df[col] == 1)
        
        rule_score[has_sepsis] += 0.5
        
        # Factor 3: Age > 80 (0.2)
        if 'anchor_age' in df.columns:
            is_old = df['anchor_age'] > 80
            rule_score[is_old] += 0.2
            
        # Factor 4: High WBC (51301) > 15 (0.3)
        # WBC > 15 is moderate leukocytosis
        wbc_col = 'lab_51301_max'
        if wbc_col in df.columns:
            high_wbc = df[wbc_col] > 15
            rule_score[high_wbc] += 0.3
        
        # Cap score
        rule_score = np.minimum(1.0, rule_score)
        
        # Weighted Ensemble
        final_prob = (1 - weight) * y_prob + weight * rule_score
        
        return final_prob

class ConformalLayer:
    """
    Simulates Conformal Prediction.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.scores = []

    def calibrate(self, y_prob, y_true):
        # Simple split conformal prediction for classification (using 1 - prob of true class)
        # For binary: score = 1 - prob_correct_class
        # if y=1, score = 1 - p. if y=0, score = 1 - (1-p) = p.
        scores = []
        for p, y in zip(y_prob, y_true):
            if y == 1:
                scores.append(1 - p)
            else:
                scores.append(p) # 1 - (1-p)
        
        self.scores = np.sort(scores)
        self.q = np.quantile(self.scores, 1 - self.alpha)

    def predict_interval(self, y_prob):
        # Prediction set: {y | score(x, y) <= q}
        # y=1 in set if 1-p <= q => p >= 1-q
        # y=0 in set if p <= q
        sets = []
        for p in y_prob:
            s = []
            if p <= self.q:
                s.append(0)
            if p >= 1 - self.q:
                s.append(1)
            sets.append(s)
        return sets

class ClinicAlly(BaseEstimator, ClassifierMixin):
    def __init__(self, rule_weight=0.2):
        self.relational = RelationalLayer()
        self.causal = CausalLayer()
        self.symbolic = SymbolicLayer()
        self.conformal = ConformalLayer()
        self.feature_names = None
        self.rule_weight = rule_weight

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        
        # 1. Relational Mapping (Train Neural Network)
        self.relational.fit(X, y)
        
        # 2. Causal Inference (Identify drivers)
        self.causal.fit(X, y, feature_names)
        
        # 3. Calibration for Conformal (using a subset of training data usually, 
        # but here we just use the whole set for simplicity/demonstration)
        y_prob = self.relational.model.predict_proba(X)[:, 1]
        self.conformal.calibrate(y_prob, y)
        
        return self

    def predict_proba(self, X):
        # 1. Relational
        y_prob = self.relational.model.predict_proba(X)[:, 1]
        
        # 2. Symbolic (Adjust probabilities)
        if self.feature_names is not None:
            y_prob_adj = self.symbolic.apply_rules(X, y_prob, self.feature_names, weight=self.rule_weight)
        else:
            y_prob_adj = y_prob
        
        return np.vstack([1 - y_prob_adj, y_prob_adj]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
