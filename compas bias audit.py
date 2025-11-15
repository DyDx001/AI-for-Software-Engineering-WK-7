import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import ClassificationMetric

def run_compas_audit():
    print("--- Starting COMPAS Dataset Bias Audit ---")

    # 1. Load the dataset using the AIF360 loader
    # This loader has pre-processed the data for us.
    # 'race' = 1 is Caucasian (Privileged)
    # 'race' = 0 is African-American (Unprivileged)
    dataset_orig = CompasDataset()

    # Define the groups for our analysis
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

    # 2. Extract Predictions (COMPAS Score) vs. Ground Truth (Did Recidivate)
    
    # 'dataset_orig' holds the ground truth (did they actually re-offend?)
    # 'dataset_orig.labels' == 1 means "did recidivate"
    
    # We must create a *copy* of the dataset to hold the *predictions*.
    dataset_pred = dataset_orig.copy()

    # The "prediction" is the COMPAS 'decile_score'
    # We find the index of the 'decile_score' feature
    score_index = dataset_orig.feature_names.index('decile_score')
    
    # Get all the scores
    scores = dataset_orig.features[:, score_index]

    # Binarize the score. ProPublica's analysis considered a score > 4
    # as a "high-risk" prediction (1).
    # This is our model's 'y_pred'.
    y_pred_binary = (scores > 4).astype(float).reshape(-1, 1)

    # Overwrite the labels in our *prediction* dataset with this new array
    dataset_pred.labels = y_pred_binary

    # 3. Use AIF360 to Calculate Fairness Metrics
    # We compare the (ground_truth_dataset) vs (prediction_dataset)
    metric = ClassificationMetric(dataset_orig, 
                                  dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)

    # 4. Get False Positive Rates (FPR)
    # FPR = FP / (FP + TN)
    # This is the rate of people who DID NOT re-offend (TN + FP)
    # but were INCORRECTLY flagged as high-risk (FP).
    
    fpr_unprivileged = metric.false_positive_rate(privileged=False)
    fpr_privileged = metric.false_positive_rate(privileged=True)
    
    print("\n--- Audit Findings ---")
    print(f"False Positive Rate (FPR) for Unprivileged Group (Black): {fpr_unprivileged*100:.1f}%")
    print(f"False Positive Rate (FPR) for Privileged Group (White):   {fpr_privileged*100:.1f}%")

    # 5. Generate Visualization
    generate_plot(fpr_unprivileged, fpr_privileged)

def generate_plot(fpr_unprivileged, fpr_privileged):
    """Generates and saves a bar chart of the FPR disparity."""
    
    groups = ['Unprivileged (Black)', 'Privileged (White)']
    rates = [fpr_unprivileged, fpr_privileged]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(groups, rates, color=['#d9534f', '#5bc0de'])
    
    ax.set_ylabel('False Positive Rate (FPR)')
    ax.set_title('Racial Bias in COMPAS Risk Scores')
    ax.set_ylim(0, 1.0)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                f'{height*100:.1f}%',
                ha='center', va='bottom', color='white', fontweight='bold')
    
    plt.savefig("compas_bias_report.png")
    print("\nVisualization saved to 'compas_bias_report.png'")
    # To display in a notebook, you would just run:
    # plt.show()

if __name__ == "__main__":
    run_compas_audit()
