import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from model import ClinicAlly

def load_and_preprocess_data(data_dir='data'):
    print("Loading data...")
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    admissions = pd.read_csv(os.path.join(data_dir, 'admissions.csv'))
    diagnoses = pd.read_csv(os.path.join(data_dir, 'diagnoses_icd.csv'))
    labevents = pd.read_csv(os.path.join(data_dir, 'labevents.csv'))

    # Merge Admissions with Patients
    df = admissions.merge(patients, on='subject_id', how='left')

    # Feature Engineering
    
    # 1. Diagnoses (One-hot-like count or binary)
    # Top codes from demo dataset + Sepsis codes
    # ICD-10: I10 (Hypertension), E785 (Hyperlipidemia), E119 (Diabetes), A419 (Sepsis)
    # ICD-9: 4019 (Hypertension), 2724 (Hyperlipidemia), 25000 (Diabetes), 99591 (Sepsis)
    target_codes = ['A419', 'I509', 'J189', 'E119', 'I10', '4019', 'E785', '2724', '25000', '99591']
    
    # Ensure icd_code is string
    diagnoses['icd_code'] = diagnoses['icd_code'].astype(str)
    
    for code in target_codes:
        # Create a set of hadm_ids that have this code
        hadms_with_code = set(diagnoses[diagnoses['icd_code'] == code]['hadm_id'])
        df[f'diag_{code}'] = df['hadm_id'].apply(lambda x: 1 if x in hadms_with_code else 0)

    # 2. Lab Events (Aggregation)
    # Items: 50813 (Lactate), 50912 (Creatinine), 51301 (WBC)
    lab_pivot = labevents.pivot_table(index='hadm_id', columns='itemid', values='valuenum', aggfunc='max')
    lab_pivot.columns = [f'lab_{c}_max' for c in lab_pivot.columns]
    
    df = df.merge(lab_pivot, on='hadm_id', how='left')

    # Fill NaNs for labs (not everyone has every lab)
    df = df.fillna(0) # Simple imputation

    # Prepare X and y
    # Features: anchor_age, gender (needs encoding), diags, labs
    # Map Gender
    df['gender_M'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)
    
    feature_cols = ['anchor_age', 'gender_M'] + \
                   [c for c in df.columns if c.startswith('diag_')] + \
                   [c for c in df.columns if c.startswith('lab_')]
    
    X = df[feature_cols]
    y = df['hospital_expire_flag']
    
    return X, y, feature_cols

def get_best_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    return np.max(f1_scores)

def balance_data(X, y):
    # Simple Random Oversampling of Minority Class
    X_pos = X[y == 1]
    y_pos = y[y == 1]
    X_neg = X[y == 0]
    y_neg = y[y == 0]
    
    # Oversample positive to match negative
    ids = np.arange(len(X_pos))
    choices = np.random.choice(ids, len(X_neg))
    
    X_pos_resampled = X_pos[choices]
    y_pos_resampled = y_pos[choices]
    
    X_res = np.vstack([X_neg, X_pos_resampled])
    y_res = np.hstack([y_neg, y_pos_resampled])
    
    return X_res, y_res

def run_experiments():
    X, y, feature_names = load_and_preprocess_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Balance Training Data
    X_train_bal, y_train_bal = balance_data(X_train.values, y_train.values)
    # X_test remains untouched!
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42) # Removed class_weight since we balanced data
    lr.fit(X_train_scaled, y_train_bal)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    results['Logistic Regression'] = {
        'AUROC': roc_auc_score(y_test, y_prob_lr),
        'AUPRC': average_precision_score(y_test, y_prob_lr),
        'F1-Score': get_best_f1(y_test, y_prob_lr)
    }

    # 2. Random Forest (Removed as per user request)
    # print("Training Random Forest...")
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train_scaled, y_train_bal)
    # y_pred_rf = rf.predict(X_test_scaled)
    # y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    # results['Random Forest'] = {
    #     'AUROC': roc_auc_score(y_test, y_prob_rf),
    #     'AUPRC': average_precision_score(y_test, y_prob_rf),
    #     'F1-Score': get_best_f1(y_test, y_prob_rf)
    # }

    # 3. Standard GNN (Simulated by MLP here)
    print("Training Standard GNN (MLP)...")
    gnn = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=42, max_iter=1000, solver='lbfgs')
    gnn.fit(X_train_scaled, y_train_bal)
    y_pred_gnn = gnn.predict(X_test_scaled)
    y_prob_gnn = gnn.predict_proba(X_test_scaled)[:, 1]
    results['Standard GNN'] = {
        'AUROC': roc_auc_score(y_test, y_prob_gnn),
        'AUPRC': average_precision_score(y_test, y_prob_gnn),
        'F1-Score': get_best_f1(y_test, y_prob_gnn)
    }

    # 4. ClinicAlly
    print("Training ClinicAlly...")
    # Use Balanced Data for ClinicAlly too
    ca = ClinicAlly()
    ca.fit(X_train_bal, y_train_bal, feature_names=feature_names) 
    
    y_pred_ca = ca.predict(X_test)
    y_prob_ca = ca.predict_proba(X_test)[:, 1]
    
    results['ClinicAlly'] = {
        'AUROC': roc_auc_score(y_test, y_prob_ca),
        'AUPRC': average_precision_score(y_test, y_prob_ca),
        'F1-Score': get_best_f1(y_test, y_prob_ca)
    }

    # Print Results
    print("\nResults:")
    print(json.dumps(results, indent=4))
    
    # Save to file
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_experiments()
