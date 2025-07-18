#!/usr/bin/env python
"""Process the 'christopher/oeis' dataset, transforming it into a benchmark-ready Parquet file.

It orchestrates the following data transformations:
- Loads the 'train' split of the specified dataset.
- Truncates sequences to a maximum length defined by `MAX_SEQUENCE_LENGTH`.
- Filters out sequences shorter than `MIN_SEQUENCE_LENGTH`.
- Separates each sequence into `sequence_first_terms` (all but the last element)
  and `sequence_next_term` (the last element).
- Removes rows where the `sequence_first_terms` are duplicates.
- Adds a binary `is_easy` column based on the presence of the 'easy' keyword
  in the original `keywords` list.

The final processed dataset is saved as a Parquet file, named by `OUTPUT_FILE_NAME`,
within the './data/' directory.

Configuration constants for the processing pipeline are defined within this module.
"""

from src.process import process_oeis_dataset

# --- Configuration Constants ---
DATASET_NAME = "christopher/oeis"
MAX_SEQUENCE_LENGTH = 20
MIN_SEQUENCE_LENGTH = 8
OUTPUT_FILE_NAME = "oeis_benchmark.parquet"


if __name__ == "__main__":
    process_oeis_dataset(
        dataset_name=DATASET_NAME,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        min_seq_len=MIN_SEQUENCE_LENGTH,
        output_file_name=OUTPUT_FILE_NAME,
    )
