import os
import sys
import yaml
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, 
    f1_score, roc_auc_score, recall_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)

def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read YAML file and return contents as dictionary."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error reading YAML file: {e}")
        raise e

def create_directories(path_list: List[str]) -> None:
    """Create directories from a list of paths."""
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory: {path}")

def save_object(file_path: str, obj: Any) -> None:
    """Save object as pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise e

def load_object(file_path: str) -> Any:
    """Load object from pickle file."""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise e

def setup_logging(log_dir: str, log_file: str) -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Evaluate model performance and return metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                             average='macro', multi_class='ovr')
        except ValueError:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    return metrics