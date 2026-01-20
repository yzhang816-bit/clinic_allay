import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from run_experiments import load_and_preprocess_data, balance_data, get_best_f1

def optimize():
    X, y, feature_names = load_and_preprocess_data()
    X = X.values
    y = y.values
    
    # Stratified K-Fold to be robust
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search for weight
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {w: {'f1': [], 'auroc': [], 'auprc': []} for w in weights}
    
    print("Starting optimization...")
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Balance
        X_train_bal, y_train_bal = balance_data(X_train, y_train)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)
        
        # Train MLP
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), solver='lbfgs', max_iter=1000, random_state=42)
        mlp.fit(X_train_scaled, y_train_bal)
        y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate Rule Score
        df_test = pd.DataFrame(X_test, columns=feature_names)
        rule_score = np.zeros(len(X_test))
        
        # Rule Logic (Replicating model.py)
        if 'lab_50813_max' in df_test.columns:
            vals = df_test['lab_50813_max'].values
            rule_score += np.clip((vals - 1.5) / 4.0, 0, 0.8)
            
        sepsis_cols = ['diag_A419', 'diag_99591']
        has_sepsis = np.zeros(len(X_test), dtype=bool)
        for col in sepsis_cols:
            if col in df_test.columns:
                has_sepsis |= (df_test[col] == 1)
        rule_score[has_sepsis] += 0.5
        
        if 'anchor_age' in df_test.columns:
            is_old = df_test['anchor_age'] > 80
            rule_score[is_old] += 0.2
            
        wbc_col = 'lab_51301_max'
        if wbc_col in df_test.columns:
            high_wbc = df_test[wbc_col] > 15
            rule_score[high_wbc] += 0.3
            
        rule_score = np.minimum(1.0, rule_score)
        
        # Evaluate Weights
        for w in weights:
            final_prob = (1 - w) * y_prob_mlp + w * rule_score
            
            f1 = get_best_f1(y_test, final_prob)
            try:
                auroc = roc_auc_score(y_test, final_prob)
            except:
                auroc = 0.5
            auprc = average_precision_score(y_test, final_prob)
            
            results[w]['f1'].append(f1)
            results[w]['auroc'].append(auroc)
            results[w]['auprc'].append(auprc)

    print("\nResults (Average metrics):")
    print(f"{'Weight':<10} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
    for w in weights:
        avg_f1 = np.mean(results[w]['f1'])
        avg_auroc = np.mean(results[w]['auroc'])
        avg_auprc = np.mean(results[w]['auprc'])
        print(f"{w:<10} {avg_auroc:.4f}     {avg_auprc:.4f}     {avg_f1:.4f}")

if __name__ == "__main__":
    optimize()
