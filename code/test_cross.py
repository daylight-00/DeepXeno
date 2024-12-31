# test.py

import sys, os
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import argparse
import pandas as pd
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            X, Y = batch[:-1], batch[-1]
            X = [x.to(device) for x in X]
            Y = Y.to(device).cpu().numpy()
            y_pred = model(*X).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(Y)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_targets, all_preds

def calculate_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def calculate_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

def aggregate_metrics(metrics_list):
    """Aggregate metrics across folds."""
    roc_aucs = [m['roc_auc'] for m in metrics_list]
    pr_aucs = [m['pr_auc'] for m in metrics_list]
    
    metrics_summary = {
        'ROC_AUC_Mean': np.mean(roc_aucs),
        'ROC_AUC_Std': np.std(roc_aucs),
        'PR_AUC_Mean': np.mean(pr_aucs),
        'PR_AUC_Std': np.std(pr_aucs)
    }
    return metrics_summary

def plot_average_curve(curves, metric='ROC', plot_path='plots', model_name='model'):
    """
    Plots the average ROC or PR curve with standard deviation shading.
    `curves` should be a list of dictionaries with 'x' and 'y' keys.
    """
    plt.figure(figsize=(8, 6))
    all_x = []
    all_y = []
    for curve in curves:
        all_x.append(curve['x'])
        all_y.append(curve['y'])
    
    # Interpolate curves to a common set of points
    mean_x = np.linspace(0, 1, 100)
    interp_y = []
    for x, y in zip(all_x, all_y):
        interp = np.interp(mean_x, x, y)
        interp_y.append(interp)
    
    interp_y = np.array(interp_y)
    mean_y = np.mean(interp_y, axis=0)
    std_y = np.std(interp_y, axis=0)
    
    plt.plot(mean_x, mean_y, color='lightseagreen', lw=1.5, label=f'Average {metric} curve')
    plt.fill_between(mean_x, mean_y - std_y, mean_y + std_y, color='lightseagreen', alpha=0.2, label='±1 Std Dev')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate' if metric == 'ROC' else 'Recall')
    plt.ylabel('True Positive Rate' if metric == 'ROC' else 'Precision')
    plt.title(f'Average {metric} Curve ({model_name})')
    plt.legend(loc="lower right" if metric == 'ROC' else "lower left")
    
    save_path = os.path.join(plot_path, f'average_{metric.lower()}_curve-{model_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Average {metric} curve saved as {save_path}\n")

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)
    
    # Data loading
    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data']['test_path'],
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }
    
    # Initialize DataProvider and Dataset
    data_provider = DataProvider(**DATA_PROVIDER_ARGS)
    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"].get("batch_size", len(dataset))
    num_workers = config["Data"]["num_workers"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_true = np.array([sample[-1] for sample in data_provider.samples])  # Optimized data access
    
    # Handle k-fold cross-validation
    num_folds = config.get("CrossValidation", {}).get("num_folds", 1)
    metrics_list = []
    roc_curves = []
    pr_curves = []
    
    for fold in range(1, num_folds + 1):
        print(f"\n=== Evaluating Fold {fold}/{num_folds} ===")
        
        # Define model path for the current fold
        model_path = os.path.join(config["chkp_path"], f"{model_name}_fold{fold}-best.pt")
        if not os.path.exists(model_path):
            print(f"Checkpoint for fold {fold} not found at {model_path}. Skipping this fold.")
            continue
        
        # Model loading
        model = config["model"](**config["model_args"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        print(f'Model for Fold {fold} loaded on {device}')
        
        # Evaluate the model
        targets, preds = test_model(model, dataloader, device)
        roc_fpr, roc_tpr, roc_auc_score = calculate_roc_auc(targets, preds)
        precision, recall, pr_auc_score = calculate_pr_auc(targets, preds)
        
        # Store metrics
        metrics = {
            'fold': fold,
            'roc_auc': roc_auc_score,
            'pr_auc': pr_auc_score
        }
        metrics_list.append(metrics)
        print(f"Fold {fold} - ROC AUC: {roc_auc_score:.4f}, PR AUC: {pr_auc_score:.4f}")
        
        # Store curves for aggregation
        roc_curves.append({'x': roc_fpr, 'y': roc_tpr})
        pr_curves.append({'x': recall, 'y': precision})
    
    if num_folds > 1 and metrics_list:
        # Aggregate metrics
        metrics_summary = aggregate_metrics(metrics_list)
        print("\n=== Cross-Validation Metrics Summary ===")
        print(f"Average ROC AUC: {metrics_summary['ROC_AUC_Mean']:.4f} ± {metrics_summary['ROC_AUC_Std']:.4f}")
        print(f"Average PR AUC: {metrics_summary['PR_AUC_Mean']:.4f} ± {metrics_summary['PR_AUC_Std']:.4f}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv_path = os.path.join(plot_path, f'crossval_metrics-{model_name}.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Cross-validation metrics saved to {metrics_csv_path}\n")
        
        # Plot aggregated ROC and PR curves
        plot_average_curve(roc_curves, metric='ROC', plot_path=plot_path, model_name=model_name)
        plot_average_curve(pr_curves, metric='PR', plot_path=plot_path, model_name=model_name)
    
    else:
        # Single model evaluation (no cross-validation)
        if metrics_list:
            metrics = metrics_list[0]
            print("\n=== Single Model Evaluation ===")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(roc_fpr, roc_tpr, lw=1.5, label=f'ROC curve (area = {metrics["roc_auc"]:.2f})', color='lightseagreen')
            plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (All HLAs)')
            plt.legend(loc="lower right")
            save_path = os.path.join(plot_path, f'roc_curve-{model_name}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"ROC curve saved as {save_path}\n")
            
            # Plot PR curve
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, lw=1.5, label=f'PR curve (area = {metrics["pr_auc"]:.2f})', color='purple')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve (All HLAs)')
            plt.legend(loc="lower left")
            save_path = os.path.join(plot_path, f'pr_curve-{model_name}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Precision-Recall curve saved as {save_path}\n")
    
    # Optionally, handle per-HLA evaluations if needed
    # This can be integrated similarly by iterating over HLAs and aggregating metrics
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model(s) with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()

    main(args.config_path)
