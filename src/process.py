"""Process the 'christopher/oeis' dataset from the Hugging Face Hub.

Performs a series of data transformations on the 'train' split, including:
- Truncating sequences to a maximum length of 20 elements.
- Filtering out sequences shorter than 8 elements.
- Separating each sequence into its 'sequence_first_terms' (all but the last element)
  and 'sequence_next_term' (the last element).
- Removing rows where the 'sequence_first_terms' are duplicates.
- Adding a binary 'is_easy' column based on the presence of the 'easy' keyword
  in the 'keywords' list.

The final processed dataset is saved as a Parquet file named 'oeis_benchmark.parquet'
within the './data/' directory.
"""

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from src.utils import (
    add_is_easy_column,
    drop_duplicate_sequence_beginnings,
    extract_next_term,
    filter_by_min_length,
    truncate_sequence,
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Main function ---
def process_oeis_dataset(
    dataset_name: str,
    max_seq_len: int,
    min_seq_len: int,
    output_file_name: str,
) -> None:
    """Load, process, and saves the OEIS dataset from Hugging Face Hub.

    This function orchestrates the entire data processing pipeline:
    1. Loads the specified dataset and targets a specific split.
    2. Selects relevant columns.
    3. Applies sequence truncation.
    4. Filters sequences by minimum length.
    5. Extracts the next term and creates a 'sequence_first_terms' column.
    6. Drops rows with duplicate 'sequence_first_terms'.
    7. Adds an 'is_easy' flag based on keywords.
    8. Saves the final processed dataset to a Parquet file.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub (e.g., "christopher/oeis").
        max_seq_len (int): The maximum number of elements to retain in each sequence.
        min_seq_len (int): The minimum required length for sequences to be kept.
        output_file_name (str): The name of the output Parquet file.

    """
    logger.info("Starting dataset processing for '%s'...", dataset_name)

    # 1. Load the dataset
    try:
        logger.info("Attempting to load dataset '%s'...", dataset_name)
        full_ds = load_dataset(dataset_name)

        # Handle cases where the dataset is a DatasetDict or a single Dataset
        if isinstance(full_ds, DatasetDict):
            ds = full_ds["train"]
            logger.info("Successfully loaded '%s' split.", "train")
        elif isinstance(full_ds, Dataset):
            ds = full_ds  # Assume it's already the desired dataset if not a dict
            logger.warning(
                "Dataset '%s' is a single Dataset, not a DatasetDict. Processing it directly.",
                dataset_name,
            )
        else:
            logger.error(
                "Unexpected dataset type for '%s': %s. Expected Dataset or DatasetDict.",
                dataset_name,
                type(full_ds),
            )
            return

    except Exception:
        logger.exception("Failed to load dataset '%s'.", dataset_name)
        return

    # 2. Filter columns
    logger.info("Filtering columns to 'sequence_id', 'sequence_name', 'sequence', 'keywords'...")

    required_columns = ["sequence_id", "sequence_name", "sequence", "keywords"]
    columns_to_select = [col for col in required_columns if col in ds.column_names]

    if len(columns_to_select) < len(required_columns):
        missing_columns = set(required_columns) - set(columns_to_select)

        logger.warning(
            "Some required columns were not found. Missing: %s. Proceeding with columns: %s",
            list(missing_columns),
            columns_to_select,
        )

    sub_ds = ds.select_columns(columns_to_select)
    logger.info("Columns filtered successfully.")

    # 3. Truncate sequences
    if "sequence" in sub_ds.column_names:
        logger.info("Truncating sequences to a maximum length of %d...", max_seq_len)
        sub_ds = sub_ds.map(truncate_sequence, batched=True, fn_kwargs={"max_el": max_seq_len})
        logger.info("Sequences truncated successfully.")
    else:
        logger.warning("Skipping sequence truncation: 'sequence' column not found in dataset.")

    # 4. Remove sequences with less than min_seq_len elements
    if "sequence" in sub_ds.column_names:
        logger.info("Removing sequences with less than %d elements...", min_seq_len)
        sub_ds = sub_ds.filter(filter_by_min_length, batched=True, fn_kwargs={"n_el": min_seq_len})
        logger.info("Sequences with less than specified elements successfully removed.")
    else:
        logger.warning("Skipping minimum length filter: 'sequence' column not found in dataset.")

    # 5. Add sequence beginning and next term columns
    if "sequence" in sub_ds.column_names:
        logger.info("Adding 'sequence_first_terms' and 'sequence_next_term' columns...")
        sub_ds = sub_ds.map(extract_next_term, batched=True)
        sub_ds = sub_ds.remove_columns(["sequence"])
        logger.info(
            "New columns 'sequence_first_terms' and 'sequence_next_term' added successfully.",
        )
    else:
        logger.warning("Skipping extraction of next term: 'sequence' column not found in dataset.")

    # 6. Drop rows with duplicate sequence beginnings
    if "sequence_first_terms" in sub_ds.column_names:
        logger.info("Dropping sequences with duplicate beginnings...")
        sub_ds = drop_duplicate_sequence_beginnings(sub_ds)
        logger.info("Sequences with duplicate beginnings successfully dropped.")
    else:
        logger.warning(
            "Skipping duplicate drop: 'sequence_first_terms' column not found in dataset.",
        )

    # 7. Add 'is_easy' column
    if "keywords" in sub_ds.column_names:
        logger.info("Adding the 'is_easy' column...")
        sub_ds = add_is_easy_column(sub_ds)
        sub_ds = sub_ds.remove_columns(["keywords"])
        logger.info("'is_easy' column successfully added.")
    else:
        logger.warning(
            "Skipping 'is_easy' column addition: 'keywords' column not found in dataset.",
        )

    # 8. Save the resulting dataset
    base_output_dir = Path("./data")
    output_full_path = base_output_dir / output_file_name

    logger.info(
        "Final dataset size: %d rows and %d columns.",
        len(sub_ds),
        len(sub_ds.column_names),
    )
    logger.info("Attempting to save processed dataset to '%s'...", output_full_path)

    try:
        # Ensure output directory exists before saving
        output_full_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured output directory exists: %s", output_full_path.parent)

        sub_ds.to_parquet(str(output_full_path))
        logger.info("Dataset successfully saved to %s", output_full_path)
    except Exception:
        logger.exception("Failed to save dataset to '%s'.", output_full_path)
