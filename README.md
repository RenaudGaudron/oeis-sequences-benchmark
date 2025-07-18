# OEIS Next Term Prediction Dataset

This repository provides a streamlined tool for **preparing a benchmark dataset** to evaluate AI models on their ability to predict the next term in integer sequences from the Online Encyclopedia of Integer Sequences (OEIS). It also hosts the dataset itself.

---
## Table of Contents

* Introduction
* Features
* Dataset Preparation 
* Dataset Schema
* How to Use (Dataset)
* License
* OEIS End-User License
* Acknowledgements

---
## Introduction

Evaluating AI models on various benchmarks is crucial for understanding their capabilities and limitations. 

This project focuses on preparing a new benchmark dataset for **sequence prediction**, leveraging the vast database of integer sequences from the Online Encyclopedia of Integer Sequences (OEIS). This dataset aims to facilitate the assessment of an LLM's ability to infer patterns and predict the subsequent term in a given sequence.

The created dataset includes sequences categorised for two distinct benchmark difficulties: an "**easy**" benchmark and a "**regular**" benchmark. For each sequence, the task is to predict the 20th term, or the final term if the sequence is shorter than 20 elements, given its initial terms. 

This project provides an easy-to-use script to generate this OEIS evaluation dataset.

---
## Features

This project provides an easy-to-use and standardised framework for **generating a dataset** suitable for evaluating AI models on the OEIS sequence prediction task. 

It aims to streamline the dataset preparation process, allowing users to efficiently create the benchmark data for assessing model performance across the "**easy**" and "**regular**" OEIS benchmark sets. 

By offering a clear and reproducible methodology for dataset generation, this project facilitates comparative analysis of AI models to better understand their capabilities in pattern recognition and sequence completion.

---
## Dataset Preparation

The dataset used for this benchmark is derived from the `christopher/oeis` dataset available on the Hugging Face Hub. 

The `process_oeis_dataset` function, found in this repository, orchestrates a series of transformations to prepare the final benchmark dataset.

Here's a detailed breakdown of the processing steps:

1.  **Loading the Dataset**: The `process_oeis_dataset` function first loads the `christopher/oeis` dataset, specifically targeting the 'train' split.
2.  **Column Selection**: Only the essential columns are selected from the raw dataset.
3.  **Sequence Truncation**: Each sequence is truncated to a maximum length of 20 terms. 
4.  **Minimum Length Filtering**: Sequences shorter than 8 elements are filtered out. This guarantees that models always receive a sufficiently long prefix (at least 7 terms) to make a prediction for the next term.
5.  **Extraction of `sequence_first_terms` and `sequence_next_term`**: For each sequence, the last element is extracted to form the `sequence_next_term` (the target for prediction). The remaining initial elements constitute the `sequence_first_terms` (the model input). The original `sequence` column is then removed.
6.  **Duplicate Removal**: Rows with identical `sequence_first_terms` are removed to ensure the uniqueness of input sequences in the benchmark. This prevents models from being evaluated on identical inputs.
7.  **`is_easy` Flag Addition**: A boolean column named `is_easy` is added. This flag is `True` if the original OEIS entry for the sequence was tagged with the keyword "easy," indicating a lower estimated difficulty. Otherwise, it is `False`, denoting a "regular" difficulty. The original `keywords` column is then removed.
8.  **Saving the Dataset**: The final processed dataset is saved as a Parquet file named `oeis_benchmark.parquet` within the `./data/` directory of this repository.

---
## Dataset Schema

The dataset's schema is designed to provide clear inputs and targets for next-term prediction tasks. Here's a breakdown of each column:

* `sequence_id` (string): This is the **unique identifier** for each sequence, directly corresponding to its "A-number" from the OEIS database (e.g., "A000045"). It's useful for cross-referencing with the original OEIS entries.
* `sequence_name` (string): The **brief description** of the integer sequence as provided by OEIS (e.g., "Fibonacci numbers"). This gives **context** to the sequence. If included in your model's prompt, it can significantly aid in predicting the next term.
* `sequence_first_terms` (list[int]): This is the **input** for your models. It's a list of integers representing the initial terms of the sequence. These sequences have been pre-processed: they're truncated to a maximum of 19 terms, and only sequences with at least 7 terms in this list are included. This means models will always receive a sufficiently long prefix to make a prediction.
* `sequence_next_term` (int): This is the **target** for your models. It's the single integer value that represents the true next term in the sequence, following directly from `sequence_first_terms`.
* `is_easy` (bool): A boolean flag indicating the **estimated difficulty** of the sequence. `True` means the original OEIS entry for this sequence was tagged with the keyword "easy," while `False` indicates a "regular" difficulty. This allows you to evaluate model performance on different challenge levels.

---
## How to Use (Dataset)

You can easily load this dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("RenaudGaudron/oeis-sequences-benchmark")
print(dataset)
```

---
## License
MIT License.

---
## OEIS End-User License
This dataset uses data from the Online Encyclopedia of Integer Sequences (OEIS). Please refer to the OEIS End-User License for terms of use related to the source data: http://oeis.org/LICENSE

---
##  Acknowledgements
This project leverages the incredible work of various communities. We extend our sincere gratitude to:

* Hugging Face: For their platform and invaluable datasets library, which hosts and facilitates the use of this benchmark, and for their transformers library, which is central to data management and dataset creation within this repository.

* The Online Encyclopedia of Integer Sequences (OEIS): For providing the rich database of integer sequences that forms the basis of this benchmark dataset.

* Christopher Akiki: For uploading the structured OEIS dataset to Hugging Face, which is instrumental for this dataset creation tool.

* The Python Community: For the rich ecosystem of libraries that enabled the dataset's creation.
