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