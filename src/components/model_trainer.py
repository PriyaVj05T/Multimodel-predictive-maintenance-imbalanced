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
    
    def evaluate_model(self, models: Dict[str, Any], X_test: pd.DataFrame, 
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
