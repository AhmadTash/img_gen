import csv
import os
import pandas as pd
from typing import Optional

DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset.csv")

def log_sample(features: dict, params: dict, file: str = DATASET_FILE) -> None:
    """
    Appends a new sample to the dataset CSV.
    Creates the file with headers if it doesn't exist.
    """
    # Merge dictionaries for a single row
    row = {**features, **params}
    
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(file)
    
    with open(file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(row)

def load_dataset(file: str = DATASET_FILE) -> Optional[pd.DataFrame]:
    """
    Loads the dataset from CSV. Returns None if file doesn't exist.
    """
    if not os.path.isfile(file):
        return None
    
    try:
        df = pd.read_csv(file)
        return df
    except Exception:
        return None
