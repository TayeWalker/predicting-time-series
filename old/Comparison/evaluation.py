import numpy as np
import matplotlib.pyplot as plt
import argparse 

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


def load_data(uni_path, multi_path, truth_path):
    univariate_predictions = np.load(uni_path, allow_pickle = True) # List of arrays
    multivariate_predictions = np.transpose(np.load(multi_path, allow_pickle = True))  # List of arrays
    ground_truth = np.transpose(np.load(truth_path, allow_pickle = True))
    return univariate_predictions, multivariate_predictions, ground_truth

def evaluate(univariate_predictions, multivariate_predictions, ground_truth):
    # Initialize lists to store metrics
    mae_uni, mae_multi = [], []
    smape_uni, smape_multi = [], []

    # Loop over each prediction and calculate metrics
    for uni_pred, multi_pred, gt in zip(univariate_predictions, multivariate_predictions, ground_truth):
        # MAE
        mae_uni.append(mae(gt, uni_pred))
        mae_multi.append(mae(gt, multi_pred))
        
        # sMAPE
        smape_uni.append(smape(gt, uni_pred))
        smape_multi.append(smape(gt, multi_pred))
    return mae_uni, smape_uni, mae_multi, smape_multi
    
def plot(metrics):
    mae_uni, smape_uni, mae_multi, smape_multi = metrics
    metrics = ['MAE', 'sMAPE']
    uni_metrics = [mae_uni, smape_uni]
    multi_metrics = [mae_multi, smape_multi]
    series_index = np.arange(1, 10)
    # Bar plot implementation
    for i, metric in enumerate(metrics):

        plt.figure(figsize=(10, 6))
        width = 0.35  # Bar width

        x = series_index  # X-axis positions

        plt.bar(x - width / 2, uni_metrics[i], width, label=f'Univariate {metric}', color='blue')
        plt.bar(x + width / 2, multi_metrics[i], width, label=f'Multivariate {metric}', color='red')
        # Adding labels and title
        plt.title(f'{metric} Comparison')
        plt.xlabel('Series Index')
        plt.ylabel(metric)
        plt.xticks(series_index)  # Aligning x-ticks with series indices
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Adding grid for readability
        plt.tight_layout()
        plt.savefig(f"comparison/{metric}_bar_plot.png")

"Bar plots generated and saved successfully!"
if __name__ == "__main__":
    # Parse file paths for predictions and ground truth
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    parser.add_argument("--uni", required=True, help="Path to univariate predictions file (NumPy format)")
    parser.add_argument("--multi", required=True, help="Path to multivariate predictions file (NumPy format)")
    parser.add_argument("--truth", required=True, help="Path to ground truth file (NumPy format)")
    
    args = parser.parse_args()
    
    univariate, multivariate, ground_truth = load_data(args.uni, args.multi, args.truth)
    print(f"Univariate predictions shape: {univariate.shape}")
    print(f"Multivariate predictions shape: {multivariate.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    plot(evaluate(univariate, multivariate, ground_truth))
    
