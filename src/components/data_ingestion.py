import pandas as pd
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    source_path: str
    train_data_path: str
    test_data_path: str
    raw_data_path: str

class DataIngestion:
    """Handle data ingestion from source."""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """Read data from source and return train/test paths."""
        logging.info("Starting data ingestion process")
        
        try:
            # Read the dataset
            df = pd.read_csv(self.config.source_path)
            logging.info(f"Dataset shape: {df.shape}")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to: {self.config.raw_data_path}")
            
            # For now, we'll save the same data to train and test
            # The actual splitting will be done in data transformation
            df.to_csv(self.config.train_data_path, index=False, header=True)
            df.to_csv(self.config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed successfully")
            return self.config.train_data_path, self.config.test_data_path
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise e