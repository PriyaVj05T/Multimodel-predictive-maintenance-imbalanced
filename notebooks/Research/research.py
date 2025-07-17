

# ==============================================
# config/config.yaml
# ==============================================
"""
# Data Configuration
data:
  source_path: "data/predictive_maintenance.csv"
  target_column: "Target"
  failure_type_column: "Failure Type"
  drop_columns: ["UDI", "Product ID"]
  
  # Column mappings
  column_mappings:
    "Air temperature [K]": "Air temperature"
    "Process temperature [K]": "Process temperature"
    "Rotational speed [rpm]": "Rotational speed"
    "Torque [Nm]": "Torque"
    "Tool wear [min]": "Tool wear"

# Preprocessing Configuration
preprocessing:
  test_size: 0.25
  random_state: 42
  stratified_splits: 5
  
  # Scaling configuration
  robust_scaler_columns: ["Rotational speed", "Torque"]
  minmax_scaler_columns: ["Air temperature", "Process temperature", "Tool wear"]
  
  # Ordinal encoding
  type_categories: ["L", "M", "H"]

# Model Configuration
models:
  target_prediction:
    threshold: 0.4
    use_tomek_links: true
    bagging_estimators: 120
    
  failure_type_prediction:
    balanced_bagging_estimators: 13
    
  # Ensemble models for comparison
  ensemble_models:
    - "BalancedRandomForestClassifier"
    - "BalancedBaggingClassifier"
    - "RUSBoostClassifier"
    - "EasyEnsembleClassifier"

# Artifacts Configuration
artifacts:
  root_dir: "artifacts"
  data_ingestion_dir: "data_ingestion"
  data_transformation_dir: "data_transformation"
  model_trainer_dir: "model_trainer"
  
# Logging Configuration
logging:
  log_dir: "logs"
  log_file: "running_logs.log"
"""

# ==============================================
# src/utils/common.py
# ==============================================
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

# ==============================================
# src/components/data_ingestion.py
# ==============================================
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

# ==============================================
# src/components/data_transformation.py
# ==============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import TomekLinks
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import logging

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

class DataTransformation:
    """Handle data preprocessing and transformation."""
    
    def __init__(self, config: DataTransformationConfig, data_config: Dict[str, Any]):
        self.config = config
        self.data_config = data_config
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data."""
        logging.info("Starting data preprocessing")
        
        # Create a copy
        processed_df = df.copy()
        
        # Drop unnecessary columns
        drop_cols = self.data_config.get('drop_columns', [])
        processed_df.drop(drop_cols, axis=1, inplace=True)
        
        # Rename columns
        column_mappings = self.data_config.get('column_mappings', {})
        processed_df.rename(columns=column_mappings, inplace=True)
        
        return processed_df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logging.info("Encoding categorical features")
        
        # Get failure types
        failure_types = df['Failure Type'].unique().tolist()
        
        # Create ordinal encoder
        type_categories = self.data_config.get('type_categories', ['L', 'M', 'H'])
        ord_enc = OrdinalEncoder(categories=[type_categories, failure_types])
        
        # Fit and transform
        encoded_data = ord_enc.fit_transform(df[['Type', 'Failure Type']])
        
        # Drop original columns and add encoded ones
        df_encoded = df.drop(['Type', 'Failure Type'], axis=1)
        encoded_df = pd.DataFrame(encoded_data, 
                                 index=df.index, 
                                 columns=['Type', 'Failure Type'])
        
        result_df = pd.concat([df_encoded, encoded_df], axis=1)
        
        # Save encoder
        save_object(
            file_path=self.config.preprocessor_obj_file_path.replace('.pkl', '_encoder.pkl'),
            obj=ord_enc
        )
        
        return result_df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        logging.info("Scaling numerical features")
        
        df_scaled = df.copy()
        
        # Robust scaling
        robust_cols = self.data_config.get('robust_scaler_columns', [])
        if robust_cols:
            robust_scaler = RobustScaler()
            robust_scaled = robust_scaler.fit_transform(df[robust_cols])
            robust_scaled_df = pd.DataFrame(robust_scaled, columns=robust_cols)
            
            df_scaled.drop(robust_cols, axis=1, inplace=True)
            df_scaled = pd.concat([df_scaled, robust_scaled_df], axis=1)
            
            # Save scaler
            save_object(
                file_path=self.config.preprocessor_obj_file_path.replace('.pkl', '_robust_scaler.pkl'),
                obj=robust_scaler
            )
        
        # MinMax scaling
        minmax_cols = self.data_config.get('minmax_scaler_columns', [])
        if minmax_cols:
            minmax_scaler = MinMaxScaler()
            minmax_scaled = minmax_scaler.fit_transform(df[minmax_cols])
            minmax_scaled_df = pd.DataFrame(minmax_scaled, columns=minmax_cols)
            
            df_scaled.drop(minmax_cols, axis=1, inplace=True)
            df_scaled = pd.concat([df_scaled, minmax_scaled_df], axis=1)
            
            # Save scaler
            save_object(
                file_path=self.config.preprocessor_obj_file_path.replace('.pkl', '_minmax_scaler.pkl'),
                obj=minmax_scaler
            )
        
        return df_scaled
    
    def create_train_test_split(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create stratified train-test split."""
        logging.info(f"Creating train-test split for target: {target_col}")
        
        X = df.drop(['Target', 'Failure Type'], axis=1)
        y = df[target_col]
        
        # Stratified split
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=self.data_config.get('test_size', 0.25),
            random_state=self.data_config.get('random_state', 42)
        )
        
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        # Log split information
        logging.info(f"Train set size: {len(X_train)}")
        logging.info(f"Test set size: {len(X_test)}")
        logging.info(f"Target distribution in train: {y_train.value_counts(normalize=True).to_dict()}")
        logging.info(f"Target distribution in test: {y_test.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_tomek_links(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply Tomek Links undersampling."""
        logging.info("Applying Tomek Links undersampling")
        
        tomek_links = TomekLinks(n_jobs=-1)
        X_resampled, y_resampled = tomek_links.fit_resample(X_train, y_train)
        
        logging.info(f"Original training set size: {len(X_train)}")
        logging.info(f"Resampled training set size: {len(X_resampled)}")
        
        return X_resampled, y_resampled
    
    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main transformation pipeline."""
        logging.info("Starting data transformation")
        
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Preprocess
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            
            # Encode categorical features
            train_df = self.encode_categorical_features(train_df)
            
            # Scale features
            train_df = self.scale_features(train_df)
            
            # Save processed data
            train_df.to_csv(self.config.transformed_train_file_path, index=False)
            test_df.to_csv(self.config.transformed_test_file_path, index=False)
            
            logging.info("Data transformation completed successfully")
            return train_df
            
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise e

# ==============================================
# src/components/model_trainer.py
# ==============================================
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import (
    BalancedRandomForestClassifier, 
    BalancedBaggingClassifier, 
    RUSBoostClassifier, 
    EasyEnsembleClassifier
)
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import logging

@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    trained_model_file_path: str
    model_report_file_path: str

class ModelTrainer:
    """Handle model training for both target and failure type prediction."""
    
    def __init__(self, config: ModelTrainerConfig, model_config: Dict[str, Any]):
        self.config = config
        self.model_config = model_config
        
    def train_target_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train model for target prediction (failure/no failure)."""
        logging.info("Training target prediction model")
        
        # Get configuration
        target_config = self.model_config.get('target_prediction', {})
        
        # Create and train model
        model = BaggingClassifier(
            n_estimators=target_config.get('bagging_estimators', 120),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        logging.info("Target model training completed")
        
        return model
    
    def train_failure_type_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train model for failure type prediction."""
        logging.info("Training failure type prediction model")
        
        # Get configuration
        failure_config = self.model_config.get('failure_type_prediction', {})
        
        # Create and train model
        model = BalancedBaggingClassifier(
            n_estimators=failure_config.get('balanced_bagging_estimators', 13),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        logging.info("Failure type model training completed")
        
        return model
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train ensemble models for comparison."""
        logging.info("Training ensemble models")
        
        models = {
            'BalancedRandomForest': BalancedRandomForestClassifier(random_state=42, n_jobs=-1),
            'BalancedBagging': BalancedBaggingClassifier(random_state=42, n_jobs=-1),
            'RUSBoost': RUSBoostClassifier(random_state=42),
            'EasyEnsemble': EasyEnsembleClassifier(random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            logging.info(f"Training {name}")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models
    
    def evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models."""
        logging.info("Evaluating models")
        
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            results[name] = metrics
            
            logging.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                        f"F1: {metrics['f1_score']:.4f}, "
                        f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")
        
        return results
    
    def apply_threshold(self, model: Any, X_test: pd.DataFrame, 
                       threshold: float = 0.4) -> np.ndarray:
        """Apply custom threshold to predictions."""
        y_pred_proba = model.predict_proba(X_test)
        y_pred_custom = (y_pred_proba[:, 1] >= threshold).astype(int)
        return y_pred_custom
    
    def initiate_model_training(self, transformed_data: pd.DataFrame) -> Dict[str, Any]:
        """Main model training pipeline."""
        logging.info("Starting model training pipeline")
        
        try:
            results = {}
            
            # Data transformation component would have prepared the data
            data_transformer = DataTransformation(
                config=None,  # Would be passed from pipeline
                data_config=self.model_config
            )
            
            # Train target prediction model
            X_train_target, X_test_target, y_train_target, y_test_target = \
                data_transformer.create_train_test_split(transformed_data, 'Target')
            
            # Apply Tomek Links if configured
            if self.model_config.get('target_prediction', {}).get('use_tomek_links', False):
                X_train_target, y_train_target = data_transformer.apply_tomek_links(
                    X_train_target, y_train_target
                )
            
            # Train target model
            target_model = self.train_target_model(X_train_target, y_train_target)
            
            # Evaluate with custom threshold
            threshold = self.model_config.get('target_prediction', {}).get('threshold', 0.4)
            y_pred_target = self.apply_threshold(target_model, X_test_target, threshold)
            
            target_metrics = evaluate_model(y_test_target, y_pred_target)
            results['target_model'] = {
                'model': target_model,
                'metrics': target_metrics
            }
            
            # Train failure type prediction model
            X_train_failure, X_test_failure, y_train_failure, y_test_failure = \
                data_transformer.create_train_test_split(transformed_data, 'Failure Type')
            
            failure_model = self.train_failure_type_model(X_train_failure, y_train_failure)
            y_pred_failure = failure_model.predict(X_test_failure)
            
            failure_metrics = evaluate_model(y_test_failure, y_pred_failure)
            results['failure_type_model'] = {
                'model': failure_model,
                'metrics': failure_metrics
            }
            
            # Train ensemble models for comparison
            ensemble_models = self.train_ensemble_models(X_train_target, y_train_target)
            ensemble_results = self.evaluate_models(ensemble_models, X_test_target, y_test_target)
            results['ensemble_comparison'] = ensemble_results
            
            # Save best models
            save_object(
                file_path=self.config.trained_model_file_path.replace('.pkl', '_target.pkl'),
                obj=target_model
            )
            
            save_object(
                file_path=self.config.trained_model_file_path.replace('.pkl', '_failure_type.pkl'),
                obj=failure_model
            )
            
            logging.info("Model training completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e

# ==============================================
# src/pipelines/training_pipeline.py
# ==============================================
import os
import logging
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.common import read_yaml, setup_logging

class TrainingPipeline:
    """Main training pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = read_yaml(config_path)
        self.setup_logging()
        self.setup_artifact_dirs()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        setup_logging(
            log_dir=log_config.get('log_dir', 'logs'),
            log_file=log_config.get('log_file', 'training.log')
        )
        
    def setup_artifact_dirs(self):
        """Create artifact directories."""
        artifacts_config = self.config.get('artifacts', {})
        root_dir = artifacts_config.get('root_dir', 'artifacts')
        
        dirs_to_create = [
            root_dir,
            os.path.join(root_dir, artifacts_config.get('data_ingestion_dir', 'data_ingestion')),
            os.path.join(root_dir, artifacts_config.get('data_transformation_dir', 'data_transformation')),
            os.path.join(root_dir, artifacts_config.get('model_trainer_dir', 'model_trainer'))
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_training_pipeline(self):
        """Execute the complete training pipeline."""
        logging.info("Starting training pipeline")
        
        try:
            # Data Ingestion
            data_ingestion_config = DataIngestionConfig(
                source_path=self.config['data']['source_path'],
                train_data_path=os.path.join(self.config['artifacts']['root_dir'], 
                                           self.config['artifacts']['data_ingestion_dir'], 
                                           'train.csv'),
                test_data_path=os.path.join(self.config['artifacts']['root_dir'], 
                                          self.config['artifacts']['data_ingestion_dir'], 
                                          'test.csv'),
                raw_data_path=os.path.join(self.config['artifacts']['root_dir'], 
                                         self.config['artifacts']['data_ingestion_dir'], 
                                         'raw.csv')
            )
            
            data_ingestion = DataIngestion(data_ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            data_transformation_config = DataTransformationConfig(
                preprocessor_obj_file_path=os.path.join(self.config['artifacts']['root_dir'], 
                                                       self.config['artifacts']['data_transformation_dir'], 
                                                       'preprocessor.pkl'),
                transformed_train_file_path=os.path.join(self.config['artifacts']['root_dir'], 
                                                        self.config['artifacts']['data_transformation_dir'], 
                                                        'train_transformed.csv'),
                transformed_test_file_path=os.path.join(self.config['artifacts']['root_dir'], 
                                                       self.config['artifacts']['data_transformation_dir'], 
                                                       'test_transformed.csv')
            )
            
            data_transformation = DataTransformation(
                data_transformation_config, 
                self.config['preprocessing']
            )
            transformed_data = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Model Training
            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=os.path.join(self.config['artifacts']['root_dir'], 
                                                    self.config['artifacts']['model_trainer_dir'], 
                                                    'model.pkl'),
                model_report_file_path=os.path.join(self.config['artifacts']['root_dir'], 
                                                   self.config['artifacts']['model_trainer_dir'], 
                                                   'model_report.yaml')
            )
            
            model_trainer = ModelTrainer(model_trainer_config, self.config['models'])
            training_results = model_trainer.initiate_model_training(transformed_data)
            
            logging.info("Training pipeline completed successfully")
            return training_results
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise e

# ==============================================
# main.py
# ==============================================
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomDataProcessor
from src.utils.common import setup_logging
import logging

def main():
    """Main function to run the predictive maintenance system."""
    
    # Setup logging
    setup_logging("logs", "main.log")
    
    try:
        # Initialize training pipeline
        training_pipeline = TrainingPipeline()
        
        # Run training
        logging.info("Starting training process...")
        training_results = training_pipeline.run_training_pipeline()
        
        # Print training results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        
        if 'target_model' in training_results:
            target_metrics = training_results['target_model']['metrics']
            print(f"\nTarget Model (Failure Prediction):")
            print(f"  Accuracy: {target_metrics['accuracy']:.4f}")
            print(f"  Precision: {target_metrics['precision']:.4f}")
            print(f"  Recall: {target_metrics['recall']:.4f}")
            print(f"  F1-Score: {target_metrics['f1_score']:.4f}")
        
        if 'failure_type_model' in training_results:
            failure_metrics = training_results['failure_type_model']['metrics']
            print(f"\nFailure Type Model:")
            print(f"  Accuracy: {failure_metrics['accuracy']:.4f}")
            print(f"  Precision: {failure_metrics['precision']:.4f}")
            print(f"  Recall: {failure_metrics['recall']:.4f}")
            print(f"  F1-Score: {failure_metrics['f1_score']:.4f}")
        
        if 'ensemble_comparison' in training_results:
            print(f"\nEnsemble Models Comparison:")
            for model_name, metrics in training_results['ensemble_comparison'].items():
                print(f"  {model_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    F1-Score: {metrics['f1_score']:.4f}")
        
        print("\n" + "="*50)
        print("MAKING SAMPLE PREDICTIONS")
        print("="*50)
        
        # Initialize prediction pipeline
        prediction_pipeline = PredictionPipeline()
        
        # Create sample data for prediction
        sample_data = CustomDataProcessor.create_sample_data(n_samples=3)
        print(f"\nSample input data:")
        print(sample_data.to_string())
        
        # Make comprehensive predictions
        predictions = prediction_pipeline.predict_comprehensive(sample_data, threshold=0.4)
        
        # Format and display results
        formatted_output = CustomDataProcessor.format_prediction_output(predictions)
        print(f"\n{formatted_output}")
        
        logging.info("Main process completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise e

if __name__ == "__main__":
    main()



"""
SETUP INSTRUCTIONS:

1. Create the directory structure:
   mkdir -p predictive_maintenance/{config,src/{components,pipelines,utils},artifacts,logs,data}

2. Create all the Python files as shown above

3. Install requirements:
   pip install -r requirements.txt

4. Place your dataset in the data folder:
   cp predictive_maintenance.csv data/

5. Update config/config.yaml with correct paths

6. Run the training pipeline:
   python main.py

7. For custom predictions:
   from src.pipelines.prediction_pipeline import PredictionPipeline
   pipeline = PredictionPipeline()
   results = pipeline.predict_from_csv("new_data.csv")

KEY FEATURES:
- Modular architecture with clear separation of concerns
- Configuration-driven approach
- Comprehensive logging
- Support for both target and failure type prediction
- Ensemble model comparison
- Custom threshold support
- Robust error handling
- Easy to extend and maintain

CONFIGURATION:
- Modify config/config.yaml to adjust model parameters
- Change data paths, model settings, preprocessing options
- Add new models or preprocessing steps easily

PREDICTION:
- Use PredictionPipeline for real-time predictions
- Support for CSV batch processing
- Comprehensive output with confidence scores
- Validation of input data format
"""

# ==============================================
# src/pipelines/prediction_pipeline.py
# ==============================================
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from src.utils.common import load_object

class PredictionPipeline:
    """Handle predictions for new data."""
    
    def __init__(self, model_dir: str = "artifacts/model_trainer"):
        self.model_dir = model_dir
        self.target_model = None
        self.failure_type_model = None
        self.encoders = {}
        self.scalers = {}
        self.load_models_and_preprocessors()
    
    def load_models_and_preprocessors(self):
        """Load trained models and preprocessors."""
        try:
            # Load models
            self.target_model = load_object(
                os.path.join(self.model_dir, 'model_target.pkl')
            )
            self.failure_type_model = load_object(
                os.path.join(self.model_dir, 'model_failure_type.pkl')
            )
            
            # Load preprocessors
            transform_dir = "artifacts/data_transformation"
            self.encoders['ordinal'] = load_object(
                os.path.join(transform_dir, 'preprocessor_encoder.pkl')
            )
            self.scalers['robust'] = load_object(
                os.path.join(transform_dir, 'preprocessor_robust_scaler.pkl')
            )
            self.scalers['minmax'] = load_object(
                os.path.join(transform_dir, 'preprocessor_minmax_scaler.pkl')
            )
            
            logging.info("Models and preprocessors loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise e
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        logging.info("Preprocessing input data for prediction")
        
        processed_data = input_data.copy()
        
        # Apply column renaming (same as training)
        column_mappings = {
            "Air temperature [K]": "Air temperature",
            "Process temperature [K]": "Process temperature",
            "Rotational speed [rpm]": "Rotational speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool wear"
        }
        processed_data.rename(columns=column_mappings, inplace=True)
        
        # For prediction, we need to handle the case where Failure Type might not be present
        # We'll create a dummy failure type column for encoding consistency
        if 'Failure Type' not in processed_data.columns:
            processed_data['Failure Type'] = 'No Failure'  # Default value
        
        # Apply ordinal encoding
        try:
            encoded_data = self.encoders['ordinal'].transform(processed_data[['Type', 'Failure Type']])
            processed_data.drop(['Type', 'Failure Type'], axis=1, inplace=True)
            encoded_df = pd.DataFrame(encoded_data, 
                                     index=processed_data.index, 
                                     columns=['Type', 'Failure Type'])
            processed_data = pd.concat([processed_data, encoded_df], axis=1)
        except Exception as e:
            logging.error(f"Error in ordinal encoding: {e}")
            raise e
        
        # Apply robust scaling
        robust_cols = ['Rotational speed', 'Torque']
        if all(col in processed_data.columns for col in robust_cols):
            try:
                robust_scaled = self.scalers['robust'].transform(processed_data[robust_cols])
                robust_scaled_df = pd.DataFrame(robust_scaled, 
                                               index=processed_data.index, 
                                               columns=robust_cols)
                processed_data.drop(robust_cols, axis=1, inplace=True)
                processed_data = pd.concat([processed_data, robust_scaled_df], axis=1)
            except Exception as e:
                logging.error(f"Error in robust scaling: {e}")
                raise e
        
        # Apply minmax scaling
        minmax_cols = ['Air temperature', 'Process temperature', 'Tool wear']
        if all(col in processed_data.columns for col in minmax_cols):
            try:
                minmax_scaled = self.scalers['minmax'].transform(processed_data[minmax_cols])
                minmax_scaled_df = pd.DataFrame(minmax_scaled, 
                                               index=processed_data.index, 
                                               columns=minmax_cols)
                processed_data.drop(minmax_cols, axis=1, inplace=True)
                processed_data = pd.concat([processed_data, minmax_scaled_df], axis=1)
            except Exception as e:
                logging.error(f"Error in minmax scaling: {e}")
                raise e
        
        # Remove target columns if present
        columns_to_remove = ['Target', 'Failure Type']
        for col in columns_to_remove:
            if col in processed_data.columns:
                processed_data.drop(col, axis=1, inplace=True)
        
        logging.info("Input data preprocessing completed")
        return processed_data
    
    def predict_failure(self, input_data: pd.DataFrame, 
                       threshold: float = 0.4) -> Dict[str, Any]:
        """Predict failure/no failure."""
        logging.info("Predicting failure status")
        
        try:
            # Preprocess input data
            preprocessed_data = self.preprocess_input(input_data)
            
            # Get prediction probabilities
            pred_proba = self.target_model.predict_proba(preprocessed_data)
            
            # Apply threshold
            failure_probability = pred_proba[:, 1]
            predictions = (failure_probability >= threshold).astype(int)
            
            # Get confidence scores
            confidence_scores = np.max(pred_proba, axis=1)
            
            results = {
                'predictions': predictions.tolist(),
                'failure_probabilities': failure_probability.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'threshold_used': threshold
            }
            
            logging.info(f"Failure prediction completed. {np.sum(predictions)} failures predicted out of {len(predictions)} samples")
            return results
            
        except Exception as e:
            logging.error(f"Error in failure prediction: {e}")
            raise e
    
    def predict_failure_type(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict type of failure."""
        logging.info("Predicting failure type")
        
        try:
            # Preprocess input data
            preprocessed_data = self.preprocess_input(input_data)
            
            # Get predictions
            predictions = self.failure_type_model.predict(preprocessed_data)
            pred_proba = self.failure_type_model.predict_proba(preprocessed_data)
            
            # Get confidence scores
            confidence_scores = np.max(pred_proba, axis=1)
            
            # Get class labels
            class_labels = self.failure_type_model.classes_
            
            results = {
                'predictions': predictions.tolist(),
                'class_labels': class_labels.tolist(),
                'prediction_probabilities': pred_proba.tolist(),
                'confidence_scores': confidence_scores.tolist()
            }
            
            logging.info("Failure type prediction completed")
            return results
            
        except Exception as e:
            logging.error(f"Error in failure type prediction: {e}")
            raise e
    
    def predict_comprehensive(self, input_data: pd.DataFrame, 
                            threshold: float = 0.4) -> Dict[str, Any]:
        """Comprehensive prediction including both failure and failure type."""
        logging.info("Starting comprehensive prediction")
        
        try:
            # Predict failure
            failure_results = self.predict_failure(input_data, threshold)
            
            # Predict failure type
            failure_type_results = self.predict_failure_type(input_data)
            
            # Combine results
            comprehensive_results = {
                'failure_prediction': failure_results,
                'failure_type_prediction': failure_type_results,
                'summary': []
            }
            
            # Create summary for each sample
            for i in range(len(failure_results['predictions'])):
                sample_summary = {
                    'sample_index': i,
                    'will_fail': bool(failure_results['predictions'][i]),
                    'failure_probability': failure_results['failure_probabilities'][i],
                    'predicted_failure_type': failure_type_results['predictions'][i],
                    'failure_type_confidence': failure_type_results['confidence_scores'][i],
                    'overall_confidence': min(
                        failure_results['confidence_scores'][i],
                        failure_type_results['confidence_scores'][i]
                    )
                }
                comprehensive_results['summary'].append(sample_summary)
            
            logging.info("Comprehensive prediction completed")
            return comprehensive_results
            
        except Exception as e:
            logging.error(f"Error in comprehensive prediction: {e}")
            raise e
    
    def predict_from_csv(self, csv_path: str, 
                        output_path: str = None,
                        threshold: float = 0.4) -> Dict[str, Any]:
        """Predict from CSV file and optionally save results."""
        logging.info(f"Making predictions from CSV: {csv_path}")
        
        try:
            # Read CSV
            input_data = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(input_data)} samples from CSV")
            
            # Make predictions
            results = self.predict_comprehensive(input_data, threshold)
            
            # Create results DataFrame
            results_df = pd.DataFrame(results['summary'])
            
            # Add original data columns
            original_cols = input_data.columns.tolist()
            for col in original_cols:
                if col not in ['Target', 'Failure Type']:  # Exclude if present
                    results_df[f'original_{col}'] = input_data[col].values
            
            # Save results if output path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                logging.info(f"Results saved to: {output_path}")
            
            return {
                'predictions': results,
                'results_dataframe': results_df,
                'output_path': output_path
            }
            
        except Exception as e:
            logging.error(f"Error in CSV prediction: {e}")
            raise e

class CustomDataProcessor:
    """Helper class for custom data processing."""
    
    @staticmethod
    def validate_input_data(input_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data format."""
        required_columns = [
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        ]
        
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        
        if missing_columns:
            return False, missing_columns
        
        return True, []
    
    @staticmethod
    def create_sample_data(n_samples: int = 5) -> pd.DataFrame:
        """Create sample data for testing."""
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'Type': np.random.choice(['L', 'M', 'H'], n_samples),
            'Air temperature [K]': np.random.uniform(295, 305, n_samples),
            'Process temperature [K]': np.random.uniform(305, 315, n_samples),
            'Rotational speed [rpm]': np.random.uniform(1000, 2000, n_samples),
            'Torque [Nm]': np.random.uniform(20, 60, n_samples),
            'Tool wear [min]': np.random.uniform(0, 250, n_samples)
        })
        
        return sample_data
    
    @staticmethod
    def format_prediction_output(results: Dict[str, Any]) -> str:
        """Format prediction results for display."""
        output = []
        output.append("=" * 50)
        output.append("PREDICTIVE MAINTENANCE RESULTS")
        output.append("=" * 50)
        
        summary = results.get('summary', [])
        for i, sample in enumerate(summary):
            output.append(f"\nSample {i+1}:")
            output.append(f"  Will Fail: {'YES' if sample['will_fail'] else 'NO'}")
            output.append(f"  Failure Probability: {sample['failure_probability']:.3f}")
            output.append(f"  Predicted Failure Type: {sample['predicted_failure_type']}")
            output.append(f"  Confidence: {sample['overall_confidence']:.3f}")
        
        return "\n".join(output)