import os
import numpy as np
import matplotlib.pyplot as plt

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def load_data_from_folder(folder_path, truth_path):
    """Load multivariate predictions and ground truth data."""
    predictions_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')],
    )
    truth_files = sorted(
        [os.path.join(truth_path, f) for f in os.listdir(truth_path) if f.endswith('.npy')]
    )
    predictions = [np.transpose(np.load(file, allow_pickle=True)) for file in predictions_files]
    ground_truth = [np.transpose(np.load(file, allow_pickle=True)) for file in truth_files]
    return predictions_files, predictions, ground_truth

def evaluate(predictions_files, predictions, ground_truth):
    """Evaluate predictions against ground truth."""
    results = {}
    for file_name, preds, truth in zip(predictions_files, predictions, ground_truth):
        model_name = os.path.basename(file_name).replace('.npy', '')  # Use filename as model identifier
        results[model_name] = {
            "MAE": [],
            "sMAPE": []
        }
        # Loop over each prediction and calculate metrics
        for pred, gt in zip(preds, truth):
            results[model_name]["MAE"].append(mae(gt, pred))
            results[model_name]["sMAPE"].append(smape(gt, pred))
    return results

def plot(metrics, output_dir="comparison"):
    """Plot metrics for all models."""
    os.makedirs(output_dir, exist_ok=True)
    series_index = np.arange(1, len(next(iter(metrics.values()))["MAE"]) + 1)
    metric_names = ["MAE", "sMAPE"]

    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        width = 0.8 / len(metrics)  # Bar width depends on the number of models

        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            x = series_index + (i - len(metrics) / 2) * width
            plt.bar(
                x,
                model_metrics[metric_name],
                width,
                label=f"{model_name} {metric_name}",
            )

        # Adding labels and title
        plt.title(f"{metric_name} Comparison Across Models")
        plt.xlabel("Series Index")
        plt.ylabel(metric_name)
        plt.xticks(series_index)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)  # Adding grid for readability
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_name}_bar_plot.png")

    print("Bar plots generated and saved successfully!")

if __name__ == "__main__":
    # Folder containing prediction .npy files
    folder_path = "Data_tests"  # Adjust path to your folder
    ground_truth_path = "Data_tests/truth"  # Adjust path to your ground truth file

    # Load data
    predictions_files, predictions, ground_truth = load_data_from_folder(folder_path, ground_truth_path)

    #print(f"Ground truth shape: {ground_truth.shape}")

    for file in predictions_files:
        print(f"Loaded predictions: {file}")

    # Evaluate and plot results
    metrics = evaluate(predictions_files, predictions, ground_truth)
    plot(metrics, output_dir="comparison_results")
