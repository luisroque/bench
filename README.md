# Towards a Theory of Cherry-Picking in Experimental ML

## Overview

This repository contains the code and datasets used for the experiments in the paper "Towards a Theory of Cherry-Picking in Experimental ML: How to Choose the Datasets to Make Your Algorithm Shine." The paper investigates the impact of dataset selection bias, particularly the practice of cherry-picking datasets, on the evaluation and ranking of time series forecasting models. We demonstrate how selectively choosing datasets can significantly skew the perceived performance of models, potentially leading to misleading conclusions.


## Key Contributions

### 1. Cherry-Picking Framework

We propose a systematic framework to assess the impact of cherry-picking in time series forecasting evaluations. The framework involves:
- **Dataset Selection**: We compile a diverse set of benchmark datasets to represent different forecasting challenges.
- **Model Evaluation**: Models are evaluated across various dataset subsets to observe how rankings change with different dataset selections.
- **Ranking Analysis**: We track model rankings based on SMAPE (Symmetric Mean Absolute Percentage Error) and demonstrate how cherry-picking can inflate the perceived performance of certain models.

### 2. Empirical Findings

- **Impact of Cherry-Picking**: Our experiments show that cherry-picking specific datasets can lead to significant biases in model rankings. For instance, with only 4 carefully selected datasets, 54% of the models could be reported as the best performer in our setup, and 85% could be presented within the top 3.
  
- **Robustness of Models**: NHITS consistently emerged as the most robust model across various scenarios. However, we also observed that even the best-performing models exhibited considerable variability when applied to their ideal use cases.

## Repository Structure
 
- `codebase/`: Contains the core code for data loading, feature engineering, and model evaluation.
  - `evaluation/`: Code for evaluating model performance, generating plots, and assessing the impact of cherry-picking.
  - `features/`: Code for feature extraction and processing.
  - `load_data/`: Scripts for loading and preparing the benchmark datasets.

- `experiments/`: Contains scripts for running the experiments.
  - `analysis/`: Scripts for analyzing experimental results, including rank distribution and cherry-picking effects.
  - `run/`: Scripts for executing the experiments, including both classical and deep learning forecasting models.

- `requirements.txt`: Lists the Python packages required to run the code in this repository.
