from src.utils.logger import get_logger
from src.models.cnn import CNN_120dataset
from src.models.cnn import fold_cross_validation_120dataset
from src.models.pca import pca
from src.models.svm import multi_datasets, train_svm
from src.utils.labels_preparation import process_labels
from src.utils.reshape_datasets import reshape_input_eeg
from src.utils.visuals import (
    label_distribution,
    plot_binarisation_choice,
    plot_f1_scores,
    plot_grid_search_results,
    plot_heatmap,
    plot_pca,
    grid_search_trend,
)

logger = get_logger()

# Binarise labels
cohesion_binary, out_cohesion = process_labels("data/Averaged Cohesion scores.csv")

# Binary threshold visual
plot_binarisation_choice(out_cohesion)

# Pie chart label distribution
label_distribution(cohesion_binary)

# Reshape all datasets
reshape_input_eeg("data/correlations_array.csv", "data/reshaped_correlations.csv", has_part=False)
reshape_input_eeg("data/correlations_array5.csv", "data/reshaped_correlations5.csv", has_part=True)
reshape_input_eeg("data/correlations_array10.csv", "data/reshaped_correlations10.csv", has_part=True)
reshape_input_eeg("data/correlations_array60.csv", "data/reshaped_correlations60.csv", has_part=True)
reshape_input_eeg("data/correlations_array120.csv", "data/reshaped_correlations120.csv", has_part=True)

################################
# Perform SVM
best_params, best_score, results_df = train_svm("data/reshaped_correlations120.csv", "data/labels.csv")
results_df.head()

# Visuals for labels heatmap + best F1 score for SVM
heatmap_data = results_df.pivot_table(index="param_svc__kernel", columns="param_svc__gamma", values="mean_test_f1")

# Plot heatmap
# plot_heatmap(heatmap_data)
plot_data = plot_grid_search_results(results_df)
# grid_search_trend(plot_data)

################################
# Perform multiple datasets SVM
datasets = [
    ("data/reshaped_correlations.csv", "data/labels.csv"),
    ("data/reshaped_correlations10.csv", "data/labels.csv"),
    ("data/reshaped_correlations120.csv", "data/labels.csv"),
    ("data/reshaped_correlations5.csv", "data/labels.csv"),
    ("data/reshaped_correlations60.csv", "data/labels.csv"),
]

results_df = multi_datasets(datasets)
results_df  # noqa: B018

# Visuals for nultiple dataset SVM
dataset_scores = results_df.groupby("Dataset", as_index=False)["Best F1 Score"].mean()
plot_f1_scores(dataset_scores)


################################
# Perform PCA

components, features = pca("data/reshaped_correlations.csv", 15)
components_120, features_120 =pca("data/reshaped_correlations120.csv", 15)

#plot for pca
plot_pca(components, features)
plot_pca(components_120, features_120)

################################
# CNN 
fold_cross_validation_120dataset(model_class=CNN_120dataset, data_X='data/reshaped_correlations120.csv', data_Y='data/labels.csv', config_path='config/cnn_config.ini', n_splits=5)
