"""
Data Preprocessing Module
Handles loading and cleaning of insurance claims data
"""

import pandas as pd
import numpy as np
from typing import Tuple

class DataPreprocessor:
    """Preprocess insurance claims data"""

    def __init__(self, filepath: str, separator: str = ';', encoding: str = 'latin1'):
        """
        Initialize preprocessor

        Args:
            filepath: Path to CSV file
            separator: CSV delimiter
            encoding: File encoding
        """
        self.filepath = filepath
        self.separator = separator
        self.encoding = encoding
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath, sep=self.separator, encoding=self.encoding)

        # Clean column names
        self.df.columns = [col.replace('\n\n', '').strip() for col in self.df.columns]

        print(f"✓ Loaded {len(self.df)} records with {self.df.shape[1]} columns")
        return self.df

    def parse_dates(self, date_columns: list) -> None:
        """
        Parse date columns to datetime format

        Args:
            date_columns: List of column names to parse
        """
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(
                    self.df[col], 
                    format='%d/%m/%Y', 
                    errors='coerce'
                )
                print(f"✓ Parsed {col} to datetime")

    def handle_missing_values(self, strategy: str = 'fillna') -> None:
        """
        Handle missing values

        Args:
            strategy: 'fillna' or 'drop'
        """
        missing_before = self.df.isnull().sum().sum()

        if strategy == 'fillna':
            # Fill numeric columns with 0
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(0)

            # Fill categorical with 'UNKNOWN'
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            self.df[categorical_cols] = self.df[categorical_cols].fillna('UNKNOWN')

        elif strategy == 'drop':
            self.df = self.df.dropna()

        missing_after = self.df.isnull().sum().sum()
        print(f"✓ Handled {missing_before - missing_after} missing values")

    def get_data(self) -> pd.DataFrame:
        """Return processed dataframe"""
        return self.df

    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to CSV

        Args:
            output_path: Path to save file
        """
        self.df.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Quick function to load and preprocess data

    Args:
        filepath: Path to CSV file

    Returns:
        Processed DataFrame
    """
    preprocessor = DataPreprocessor(filepath)
    df = preprocessor.load_data()

    # Parse date columns
    date_cols = ['DATE SINISTRE', 'DATE DECLARATION', 'DATE VISITE TECHNIQUE VA']
    preprocessor.parse_dates(date_cols)

    # Handle missing values
    preprocessor.handle_missing_values(strategy='fillna')

    return preprocessor.get_data()
