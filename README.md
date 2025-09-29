# Replication Package:

## The Price of Precision: The Cost of Preprocessing for Automating Code Review

This replication package contains all code, data, and results necessary to reproduce the experiments and findings from the paper. The package is organized into three main folders:

- Preprocessing

- Training_and_Testing

- Results

## 1. Preprocessing

The Preprocessing folder provides the full pipelines and datasets used in the experiments. It contains the following subdirectories:

### a) Cululative_code

This directory contains the preprocessing pipeline to generate cumulative preprocessed datasets.

- The cumulative setup follows the methodology described in the paper, where preprocessing steps are applied progressively.

- To run the pipeline on any dataset and generate cumulative-level preprocessed data, you only need to update the input path in Preprocessing_Pipeline.py:

```python
input_csv_path = os.path.join(Root_dir, 'SplittedData/ProjectLevel/Gerrit_ProjectSplited_Test_NoLeakage.csv')
```

All other files in this directory can remain unchanged.

### b) Isolated_code

This directory mirrors the structure of Cululative_code, but is designed for generating datasets under the isolated setup.

- In the isolated setup, preprocessing steps are applied independently (not cumulatively), following the definitions in the paper.

- Similar to the cumulative setup, you only need to modify the input path in Preprocessing_Pipeline.py:
```python
input_csv_path = os.path.join(Root_dir, 'SplittedData/ProjectLevel/Gerrit_ProjectSplited_Test_NoLeakage.csv')
```

All other files remain unchanged.

### c) NewGitHubData 
***The content of this subfolder should be downloaded from Zenodo.***

https://doi.org/10.5281/zenodo.17227729

This directory contains the newly retrieved GitHub pull request data (after September 2021).

Because this dataset is naturally partitioned by time, it is categorized under the TimeLevel splitting.

Inside, you will find two subfolders containing preprocessed datasets generated under both:

- Cumulative setup (Cumulative_preprocesse_data)

- Isolated setup (Isolated_preprocesse_data)


### d) SplittedData
***The content of this subfolder should be downloaded from Zenodo.***

This directory provides the input datasets before preprocessing, with splits already applied.

It follows the same structure as NewGitHubData, but additionally includes project-level splitting data.

Thus, you will find:

- TimeLevel split datasets

- ProjectLevel split datasets

These serve as the starting point for running the pipelines in Cululative_code and Isolated_code.

## 2. Training_and_testing

The Training_and_testing folder contains the full experimental setup for fine-tuning and evaluating different models used in the paper. It includes four subdirectories, each corresponding to a specific model:

- CodeLLma

- CodeReviewer

- OpenNMT

- T5

Each of the four model directories (CodeLLma, CodeReviewer, OpenNMT, T5) follows a consistent layout and provides:

### a) Code 

A dir named **Code** for fine-tuning and evaluation. The replication package includes pretrained checkpoints of T5 model 
(so you don’t need to fetch them externally). CodeReviewer and CodeLLLam models models rely on pretrained weights available on Hugging Face Hub ; the code automatically pulls them from Hugging Face when you run the experiments. OpenNMTframework doesn’t ship with a pretrained checkpoint in your package — instead, the provided scripts allow you to train from scratch (using the preprocessed datasets). **Code** directory Contains the core implementation files, including the FineTuning_and_Testing.py script. FineTuning_and_Testing.py is the main driver script that:

- Fine-tunes the corresponding model on the chosen preprocessed dataset.

- Uses an early stopping strategy to avoid overfitting.

- Saves the best-performing checkpoint during training.

- Automatically evaluates the trained model on the matching test set.

- Inside this directory, you will also find the CodeXGLUE module, which is responsible for computing CodeBLEU scores for model predictions. This ensures a standardized evaluation of code quality across all models.

### b) setup.sh
- A setup.sh script that automatically generates the required configurations for all experimental setups. Running it creates folders for every combination of project vs. time-level splits and cumulative vs. isolated preprocessing. The script copies the necessary code, links the correct datasets, and updates run scripts so experiments are ready to fine-tune and evaluate directly.

## 3. Results

The Results folder contains all evaluation outputs from the experiments described in the paper. It is organized into three main parts:

### a) csvResults
***The content of this subfolder should be downloaded from Zenodo.***

- This subfolder stores raw CSV files of results for each study (S1, S2, S3) and each model used (e.g., CodeLLaMA, T5, CodeReviewer, OpenNMT, ChatGPT).  For example, csvResultsS1CodeLlama contains the results of the CodeLLaMA model fine-tuned under preprocessing configurations from Study 1.
- Each CSV records detailed evaluation metrics, (EXM (Exact Match), CodeBLEU, Levenshtein distance (similarity ratio))
- For example the file :
```python
csvResultsS1CodeLlama/S1_P1plusP3_new_test_predictions_with_codeBLEU_10_ProjectLevel.csv
```
represents results of the CodeLLaMA model, fine-tuned on P1+P3 isolated preprocessing configuration with Project-level splitting, tested on the new GitHub dataset.

### b) FinalRatio
***The content of this subfolder should be downloaded from Zenodo.***
- This subfolder consolidates performance metrics into summary tables. Results are aggregated by preprocessing step, dataset size, and model performance.

### c) StatisticalTests

This subfolder contains the  statistical significance tests. These tests assess:

- Whether differences between Project-level and Time-level splits are statistically significant.

- Whether differences between preprocessing steps within the same model are significant.

Provides the formal validation of the claims made in the paper.


