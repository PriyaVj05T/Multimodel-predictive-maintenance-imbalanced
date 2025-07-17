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
