import json
import re

def update_tex(tex_path, results_path, output_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Helper to parse string "mean ± std" -> "mean \pm std"
    # We need double backslash for latex \pm, and double again for python string escaping if passed to re.sub?
    # Actually re.sub treats backslashes in repl specially. We should use raw string or escape properly.
    def fmt(val_str):
        # We want the literal string "\pm" in the output file.
        # In re.sub replacement string, backslash needs to be escaped.
        return val_str.replace('±', r'\\pm')
    
    # Update Table 1 (Performance)
    # Rows: Logistic Regression, Random Forest, XGBoost, LSTM, RETAIN, Standard GNN, G-Net, ClinicAlly
    
    models_table1 = [
        ('Logistic Regression', 'Logistic Regression'),
        ('Random Forest', 'Random Forest'),
        ('XGBoost', 'XGBoost'),
        ('LSTM', 'LSTM'),
        ('RETAIN', 'RETAIN'),
        ('Standard GNN', 'Standard GNN'),
        ('G-Net', 'G-Net'),
        ('ClinicAlly', 'ClinicAlly (Full)')
    ]
    
    for tex_name, json_name in models_table1:
        res = results[json_name]
        # Regex to match the row
        # Pattern: Name & AUROC & AUPRC & F1 & Brier \\
        # We assume strict formatting in tex
        if tex_name == 'ClinicAlly':
             # ClinicAlly is bolded usually
            pattern = r"\\textbf\{ClinicAlly\}\s*&\s*[^&]+&\s*[^&]+&\s*[^&]+&\s*[^&]+\s*\\\\"
            new_row = f"\\textbf{{ClinicAlly}} & \\textbf{{{fmt(res['AUROC'])}}} & \\textbf{{{fmt(res['AUPRC'])}}} & \\textbf{{{fmt(res['F1'])}}} & \\textbf{{{fmt(res['Brier'])}}} \\\\"
        else:
            pattern = f"{re.escape(tex_name)}\\s*&\\s*[^&]+&\\s*[^&]+&\\s*[^&]+&\\s*[^&]+\\s*\\\\\\\\"
            new_row = f"{tex_name} & {fmt(res['AUROC'])} & {fmt(res['AUPRC'])} & {fmt(res['F1'])} & {fmt(res['Brier'])} \\\\"
            
        content = re.sub(pattern, new_row, content)

    # Update Table 2 (Ablation)
    # Rows: ClinicAlly (Full), w/o Causal Inference, w/o Symbolic Reasoning, w/o Conformal Prediction, w/o Graph Structure
    
    ablations = [
        ('ClinicAlly (Full)', 'ClinicAlly (Full)'),
        ('w/o Causal Inference', 'w/o Causal Inference'),
        ('w/o Symbolic Reasoning', 'w/o Symbolic Reasoning'),
        ('w/o Conformal Prediction', 'w/o Conformal Prediction'),
        ('w/o Graph Structure', 'w/o Graph Structure')
    ]
    
    for tex_name, json_name in ablations:
        res = results[json_name]
        pattern = f"{re.escape(tex_name)}\\s*&\\s*[^&]+&\\s*[^&]+&\\s*[^&]+&\\s*[^&]+\\s*\\\\\\\\"
        new_row = f"{tex_name} & {fmt(res['AUROC'])} & {fmt(res['AUPRC'])} & {fmt(res['F1'])} & {fmt(res['Brier'])} \\\\"
        content = re.sub(pattern, new_row, content)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Updated {output_path}")

if __name__ == "__main__":
    update_tex('v3.tex', 'results_v3.json', 'v3.tex')
