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