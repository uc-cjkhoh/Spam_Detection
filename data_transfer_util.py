import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional

from loader.config_loader import cfg
from loader.logger_loader import logging


class DataTransferUtil:
    """Utility to transfer verified rows from unlabeled to labeled data files."""
    
    def __init__(self):
        self.label_filepath = cfg.active_learning.label_data_file
        self.unlabel_filepath = cfg.active_learning.unlabel_data_file
        self.target_column = cfg.data.target_column
        
    def transfer_verified_rows(self, row_indices: Optional[List[int]] = None, 
                             verified_column: str = 'verified') -> None:
        """
        Transfer verified rows from unlabeled to labeled data file.
        
        Args:
            row_indices: Specific row indices to transfer. If None, transfers all verified rows.
            verified_column: Column name indicating verification status (True/False or 1/0)
        """
        try:
            # Load data files
            unlabel_data = pd.read_excel(self.unlabel_filepath)
            label_data = pd.read_excel(self.label_filepath)
            
            if row_indices is not None:
                # Transfer specific rows by index
                rows_to_transfer = unlabel_data.iloc[row_indices].copy()
                remaining_unlabel = unlabel_data.drop(index=row_indices).reset_index(drop=True)
            else:
                # Transfer rows marked as verified
                if verified_column not in unlabel_data.columns:
                    raise ValueError(f"Column '{verified_column}' not found. Add this column to mark verified rows.")
                
                verified_mask = unlabel_data[verified_column].isin([True, 1, 'True', 'true', 'Yes', 'yes'])
                rows_to_transfer = unlabel_data[verified_mask].copy()
                remaining_unlabel = unlabel_data[~verified_mask].reset_index(drop=True)
            
            if len(rows_to_transfer) == 0:
                logging.info("No rows to transfer.")
                return
            
            # Remove verification column if it exists
            if verified_column in rows_to_transfer.columns:
                rows_to_transfer = rows_to_transfer.drop(columns=[verified_column])
            if verified_column in remaining_unlabel.columns:
                remaining_unlabel = remaining_unlabel.drop(columns=[verified_column])
            
            # Ensure consistent column structure
            expected_columns = [self.target_column, f"{self.target_column}_label", f"{self.target_column}_score"]
            for col in expected_columns:
                if col not in rows_to_transfer.columns:
                    if col.endswith('_score'):
                        rows_to_transfer[col] = 1.0  # Max confidence for manually verified
                    elif col.endswith('_label'):
                        logging.warning(f"Missing {col}. Please ensure labels are properly set.")
                        return
            
            # Combine and save
            updated_label_data = pd.concat([label_data, rows_to_transfer], ignore_index=True)
            
            # Save updated files
            updated_label_data.to_excel(self.label_filepath, index=False)
            remaining_unlabel.to_excel(self.unlabel_filepath, index=False)
            
            logging.info(f"Successfully transferred {len(rows_to_transfer)} rows from unlabeled to labeled data.")
            
        except Exception as e:
            logging.error(f"Error during transfer: {e}")
            raise
    
    def add_verification_column(self) -> None:
        """Add a 'verified' column to the unlabeled data file for manual checking."""
        try:
            unlabel_data = pd.read_excel(self.unlabel_filepath)
            
            if 'verified' not in unlabel_data.columns:
                unlabel_data['verified'] = False
                unlabel_data.to_excel(self.unlabel_filepath, index=False)
                logging.info("Added 'verified' column to unlabeled data file.")
            else:
                logging.info("'verified' column already exists.")
                
        except Exception as e:
            logging.error(f"Error adding verification column: {e}")
            raise
    
    def preview_transfer(self, row_indices: Optional[List[int]] = None, 
                        verified_column: str = 'verified') -> None:
        """Preview which rows would be transferred without actually moving them."""
        try:
            unlabel_data = pd.read_excel(self.unlabel_filepath)
            
            if row_indices is not None:
                rows_to_transfer = unlabel_data.iloc[row_indices]
            else:
                if verified_column not in unlabel_data.columns:
                    print(f"Column '{verified_column}' not found.")
                    return
                
                verified_mask = unlabel_data[verified_column].isin([True, 1, 'True', 'true', 'Yes', 'yes'])
                rows_to_transfer = unlabel_data[verified_mask]
            
            print(f"\nRows to transfer ({len(rows_to_transfer)}):")
            print(rows_to_transfer[[self.target_column, f"{self.target_column}_label"]].to_string())
            
        except Exception as e:
            logging.error(f"Error in preview: {e}")


def main():
    parser = argparse.ArgumentParser(description='Transfer verified data between labeled and unlabeled files')
    parser.add_argument('--action', choices=['transfer', 'add-column', 'preview'], 
                       required=True, help='Action to perform')
    parser.add_argument('--indices', nargs='+', type=int, 
                       help='Specific row indices to transfer')
    parser.add_argument('--verified-column', default='verified', 
                       help='Column name for verification status')
    
    args = parser.parse_args()
    
    util = DataTransferUtil()
    
    if args.action == 'add-column':
        util.add_verification_column()
    elif args.action == 'preview':
        util.preview_transfer(args.indices, args.verified_column)
    elif args.action == 'transfer':
        util.transfer_verified_rows(args.indices, args.verified_column)


if __name__ == '__main__':
    main()
