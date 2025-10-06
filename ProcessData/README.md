# ProcessData

This folder contains scripts used for tactile gesture dataset processing: labeling, cleaning, windowing, visualization and model evaluation.


## Table of Contents

- [Labeling and block extraction](#labeling-and-block-extraction)
- [Cleaning](#cleaning)
- [Windowing generation](#windowing generation)
- [Visualization](#visualization)
- [Model alignment and performance evaluation](#model-alignment-and-performance-evaluation)
- [Utilities](#utilities)



## Labeling and block extraction

- `1_rawData2LabeledData.ipynb`：Extracts and synchronizes raw trial logs(all_data.txt,true_label.csv) into labeled_data.csv files.
- `2_labeled_data_block.ipynb` : Adds `block_id` and `touch_type` columns to labeled data and generates segmented blocks.
- `labeledDataToBlockData.ipynb`, `labeled_data_block_2.ipynb` : Alternative/extended implementations of block extraction and labeling logic.


## Cleaning

- `DeleteAbandonData_3.ipynb` : Removes abandoned/unusable rows (e.g. `block_id == -1`) from labeled datasets.


## Windowing generation

- `3_labeled_window_dataset.ipynb` : Convert block-level labeled data into overlapping windows (sliding windows). Produces windowed CSV datasets used by models.
- `6_csvTrainTestSplit.py` : Split windowed data into train and test sets by grouping on `window_id`/`block_id` to avoid leakage.


## Visualization

- `5_labelAndVisualization.ipynb` : Visual analysis of labeled data (time traces, label overlays, summary plots).
- `7_visualizeLabeld_data.ipynb` : Additional visualization helpers and examples.


## Model alignment and performance evaluation

- `4_0ModelLabelMatch_Visualization.ipynb`, `4_1ModelLabelMatch_Visualization.ipynb` : Align model outputs with ground-truth labels in time and visualize matched sequences.
- `4_1Accuracy_maryam.ipynb`, `4_1_1AccuracyPlotForPaper.ipynb` : Compute detection/gesture accuracy, delays, confusion matrices and generate publication-ready plots.
- `4_2_1ModelPerfomance.ipynb`, `4_2_ModelPerfomance_singleFolder.ipynb`, `4_3_OverallModelPerformanceVisualization.ipynb` : Aggregate performance across models and experiments and produce summary visualizations.


## Utilities 

- `Data2Models.py` : Convert CSV/windowed data into tensor/dataset objects consumable by PyTorch models (dataset classes and transforms).
- `data_obs.py` : Robot data collection script.
- `saveData.py` : Continuous data recorder. Saves raw robot streams, labels and model outputs to files for later processing.
- `old_data_obs.py` : Older data observation script.



