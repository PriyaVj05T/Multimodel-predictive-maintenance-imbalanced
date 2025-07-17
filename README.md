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