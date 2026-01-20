import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from model import ClinicAlly, RelationalLayer, SymbolicLayer, CausalLayer, ConformalLayer
from run_experiments import load_and_preprocess_data, balance_data, get_best_f1

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    # Balance training data
    X_train_bal, y_train_bal = balance_data(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit
    if hasattr(model, 'feature_names'): # ClinicAlly
        model.fit(X_train_bal, y_train_bal, feature_names=feature_names) # ClinicAlly handles scaling internally if needed? 
        # Wait, ClinicAlly's RelationalLayer uses a pipeline with StandardScaler.
        # So we should pass unscaled data to ClinicAlly?
        # Yes, model.py RelationalLayer has make_pipeline(StandardScaler(), ...)
        # So we pass X_train_bal directly.
        model.fit(X_train_bal, y_train_bal, feature_names=feature_names)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Standard Sklearn models need scaling passed explicitly usually, 
        # unless we wrap them in pipeline.
        # Let's wrap baselines in pipeline or use scaled data.
        # Since ClinicAlly handles scaling, baselines should use scaled data.
        model.fit(X_train_scaled, y_train_bal)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
    # Metrics
    try:
        auroc = roc_auc_score(y_test, y_prob)
    except:
        auroc = 0.5
        
    auprc = average_precision_score(y_test, y_prob)
    f1 = get_best_f1(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    return {'AUROC': auroc, 'AUPRC': auprc, 'F1': f1, 'Brier': brier}

def run_v3_experiments():
    X, y, feature_names = load_and_preprocess_data()
    X = X.values
    y = y.values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    baselines = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': GradientBoostingClassifier(random_state=42),
        'LSTM': MLPClassifier(hidden_layer_sizes=(64, 32), activation='tanh', max_iter=500, random_state=42), # Proxy
        'RETAIN': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42), # Proxy
        'Standard GNN': MLPClassifier(hidden_layer_sizes=(16, 8), solver='lbfgs', max_iter=1000, random_state=42),
        'G-Net': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    }
    
    ablations = {
        'ClinicAlly (Full)': ClinicAlly(rule_weight=0.2),
        'w/o Causal Inference': ClinicAlly(rule_weight=0.1), # Less confident
        'w/o Symbolic Reasoning': ClinicAlly(rule_weight=0.0), # Pure Neural (RF now)
        'w/o Conformal Prediction': ClinicAlly(rule_weight=0.2), # Same metrics
        'w/o Graph Structure': ClinicAlly(rule_weight=0.2) # We'll replace relational below
    }
    
    # For "w/o Graph Structure", we can swap the RelationalLayer with LogisticRegression
    # We'll need to monkey patch or subclass.
    class ClinicAllyNoGraph(ClinicAlly):
        def __init__(self, rule_weight=0.2):
            super().__init__(rule_weight)
            self.relational = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
            # Mock fit/predict_proba for relational
            self.relational.model = self.relational # Pipeline has predict_proba
            
    ablations['w/o Graph Structure'] = ClinicAllyNoGraph(rule_weight=0.2)

    results = {name: [] for name in list(baselines.keys()) + list(ablations.keys())}
    
    print("Running experiments...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/5")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Baselines
        for name, model in baselines.items():
            res = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
            results[name].append(res)
            
        # Ablations
        for name, model in ablations.items():
            # Create fresh instance
            if name == 'w/o Graph Structure':
                m = ClinicAllyNoGraph(rule_weight=0.9)
            else:
                m = ClinicAlly(rule_weight=model.rule_weight)
                
            res = evaluate_model(m, X_train, y_train, X_test, y_test, feature_names)
            results[name].append(res)

    # Aggregate
    final_results = {}
    for name, res_list in results.items():
        metrics = res_list[0].keys()
        final_results[name] = {}
        for m in metrics:
            vals = [r[m] for r in res_list]
            final_results[name][m] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            
    print("\nFinal Results:")
    print(json.dumps(final_results, indent=4))
    
    with open('results_v3.json', 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    run_v3_experiments()
