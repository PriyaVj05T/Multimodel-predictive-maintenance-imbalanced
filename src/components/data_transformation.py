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
        """Create stratified train-test split as data is imbalance"""
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
