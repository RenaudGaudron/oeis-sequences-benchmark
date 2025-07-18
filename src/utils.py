"""Utility functions for manipulating Hugging Face Dataset objects.

More specifically for tasks involving sequence truncation, filtering by length,
extracting sequence components, removing duplicate sequences, and adding new features
based on existing data.

The functions are designed to be compatible with `Dataset.map()` and `Dataset.filter()`
methods for efficient data processing within the Hugging Face `datasets` library.
"""

from typing import Any, Union

from datasets import Dataset


def truncate_sequence(examples: dict[str, list[Any]], max_el: int) -> dict[str, list[Any]]:
    """Truncate sequences in a batch of examples to a specified maximum number of elements.

    For each sequence in the 'sequence' column, if its length exceeds `max_el`,
    it will be truncated to include only the first `max_el` elements.
    Sequences shorter than or equal to `max_el` will remain unchanged.

    Designed for use with `Dataset.map(batched=True)`.

    Args:
        examples (dict[str, list[Any]]): A batch of examples with a 'sequence' key
                                        containing a list of sequences.
        max_el (int): The maximum number of elements to retain in each sequence.

    Returns:
        dict[str, list[Any]]: A dictionary containing the updated 'sequence' column
                             with truncated sequences.

    """
    truncated_sequences = [
        seq[:max_el] if len(seq) > max_el else seq for seq in examples["sequence"]
    ]
    return {"sequence": truncated_sequences}


def filter_by_min_length(dataset: dict[str, list[list[Any]]], n_el: int) -> list[bool]:
    """Filter a batch of dataset entries, keeping sequences with at least `n_el` elements.

    Designed for use with `Dataset.filter(batched=True)`.

    Args:
        dataset (dict[str, list[list[Any]]]): A batch of examples with a 'sequence' key
                                                containing a list of sequences.
        n_el (int): The minimum required length for sequences to be kept.

    Returns:
        list[bool]: A list of booleans (True to keep, False to discard) for each example
                    in the input batch.

    """
    return [len(seq) >= n_el for seq in dataset["sequence"]]


def extract_next_term(
    examples: dict[str, list[list[Any]]],
) -> dict[str, Union[list[list[Any]], list[Any]]]:
    """Create two new columns from column 'sequence'.

    Extract the last element from each 'sequence' into 'sequence_next_term'
    and truncates 'sequence' to 'sequence_first_terms'.

    Designed for use with `Dataset.map(batched=True)`. Assumes sequences in
    'examples["sequence"]' have at least one element.

    Args:
        examples (dict[str, list[list[Any]]]): A batch of examples with a 'sequence' key
                                                containing a list of sequences.

    Returns:
        dict[str, Union[list[list[Any]], list[Any]]]: A dictionary with two new lists:
                                                       'sequence_first_terms' (sequences
                                                       without their last element) and
                                                       'sequence_next_term' (the extracted
                                                       last elements).

    """
    sequence_first_terms = []
    sequence_next_term = []

    for seq in examples["sequence"]:
        next_term = seq[-1]
        new_sequence_beginning = seq[:-1]

        sequence_next_term.append(next_term)
        sequence_first_terms.append(new_sequence_beginning)

    return {
        "sequence_first_terms": sequence_first_terms,
        "sequence_next_term": sequence_next_term,
    }


def drop_duplicate_sequence_beginnings(dataset: Dataset) -> Dataset:
    """Drop rows from a Hugging Face Dataset where the 'sequence_beginning' list is a duplicate.

    The order of elements within the list matters for uniqueness.

    Args:
        dataset (Dataset): The input Hugging Face Dataset object.
                           It must have a 'sequence_beginning' feature where each
                           entry is a list (e.g., [1, 2, 3]).

    Returns:
        Dataset: A new Dataset object with rows containing duplicate
                 'sequence_beginning' lists removed. If the 'sequence_beginning'
                 column is not found, the original dataset is returned with an error message.

    """
    if "sequence_first_terms" not in dataset.features:
        print("Error: The 'sequence_first_terms' column was not found in the dataset.")
        return dataset

    # A set to store unique 'sequence_first_terms' lists (converted to tuples for hashability)
    seen_sequences = set()

    # The 'example' argument represents a single row (as a dictionary) from the dataset.
    def filter_unique_sequence(example: dict[str, Any]) -> bool:
        # Convert the list in 'sequence_first_terms' to a tuple.
        # Tuples are hashable, which allows them to be added to a set.
        # This ensures that the order of elements is considered for uniqueness.
        current_sequence_tuple = tuple(example["sequence_first_terms"])

        if current_sequence_tuple not in seen_sequences:
            seen_sequences.add(current_sequence_tuple)
            return True

        return False

    return dataset.filter(filter_unique_sequence)


def add_is_easy_column(dataset: Dataset) -> Dataset:
    """Add a new column 'is_easy' to a Hugging Face Dataset.

    The 'is_easy' column will contain 1 if the 'keywords' list for that row
    contains the string 'easy' (case-sensitive), and 0 otherwise.

    Args:
        dataset (Dataset): The input Hugging Face Dataset object.
                           It must have a 'keywords' feature where each entry
                           is a list of strings (e.g., ['fini', 'easy', 'new']).

    Returns:
        Dataset: A new Dataset object with the 'is_easy' column added.
                 If the 'keywords' column is not found, the original dataset
                 is returned with an error message.

    """
    if "keywords" not in dataset.features:
        print("Error: The 'keywords' column was not found in the dataset.")
        return dataset

    # Define a function to process each example (row) and determine the 'is_easy' value
    def _add_easy_flag(example: dict[str, any]) -> dict[str, int]:
        keywords: list[str] = example.get("keywords", [])
        is_easy_value = 1 if "easy" in keywords else 0
        return {"is_easy": is_easy_value}

    # `with_indices=False` is used because we don't need the row index in our transformation.
    return dataset.map(_add_easy_flag, with_indices=False)
