# METADATA [automl.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Description: This file contains the functions to train an AutoML model using H2O's AutoML.
# Developed By: 
    # Name: Vansh Raja

# Version: 1.0 [Date: 12-12-2024]
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dependencies:
    # Python 3.11.15
    # Libraries:
        # H2O 3.46.0.6
        # Pandas 2.2.3

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os

def process_data(path):
    """
    Reads a CSV file from the given path and returns the data as a pandas DataFrame.
    
    Parameters:
    path (str): The file path to the CSV file.
    
    Returns:
    pandas.DataFrame or None: The loaded data, or None if an error occurs during file reading.
    
    Key Points:
    - Uses pandas' read_csv method for data loading
    - Implements basic error handling to catch and report file reading issues
    """
    # Attempt to read the CSV file, with error handling for file access or format issues
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        # Print detailed error message if file reading fails
        print(f"Error reading the file: {e}")
        return None

def trainAutoML(train_data, target_variable, project_name=None, include_algos=None, 
                max_runtime_secs=60, max_models=20, max_runtime_secs_per_model=0, seed=1):
    """
    Trains an AutoML model using H2O's automated machine learning capabilities.
    
    Parameters:
    - train_data: H2O training frame containing the dataset
    - target_variable: The column name of the target/prediction variable
    - project_name: Optional name for the AutoML project
    - include_algos: Optional list of algorithm families to include
    - max_runtime_secs: Maximum total runtime for AutoML training (default: 60 seconds)
    - max_models: Maximum number of models to train (default: 20)
    - max_runtime_secs_per_model: Maximum time per individual model (default: no limit)
    - seed: Random seed for reproducibility (default: 1)
    
    Returns:
    Tuple containing the model leaderboard and the best performing model (leader)
    
    Key Configurations:
    - Automatically tries multiple algorithms
    - Limits total training time and number of models
    - Enables reproducible results with seed
    """
    # Initialize H2OAutoML with specified configuration parameters
    aml = H2OAutoML(
        project_name=project_name,
        include_algos=include_algos, 
        max_runtime_secs=max_runtime_secs,
        max_models=max_models, 
        max_runtime_secs_per_model=max_runtime_secs_per_model, 
        seed=seed
    )
    
    # Train the AutoML model using the specified training data and target variable
    aml.train(training_frame=train_data, y=target_variable)
    
    # Return the model leaderboard (ranking of all models) and the best model
    return aml.leaderboard, aml.leader

def get_model_ids(model_leaderboard):
    """
    Extracts model IDs from the H2O AutoML leaderboard.
    
    Parameters:
    model_leaderboard: H2O leaderboard containing trained models
    
    Returns:
    List of model IDs sorted by performance
    
    Conversion Process:
    1. Extract 'model_id' column
    2. Convert to pandas DataFrame
    3. Extract model ID list
    """
    # Extract model IDs as a list, sorted by performance
    return model_leaderboard['model_id'].as_data_frame()['model_id'].tolist()

def save_model(model_id: str, path: str):
    """
    Saves a specific H2O model to the designated file path.
    
    Parameters:
    - model_id (str): Unique identifier of the H2O model to save
    - path (str): Directory where the model will be stored
    
    Returns:
    str or None: Path to the saved model file, or None if saving fails
    
    Key Features:
    - Creates the save directory if it doesn't exist
    - Overwrites existing models with the same name
    - Provides detailed error handling
    """
    try:
        # Retrieve the specific model using its unique identifier
        model = h2o.get_model(model_id)
        
        # Ensure the target save directory exists, create if necessary
        os.makedirs(path, exist_ok=True)
        
        # Save the model, forcing overwrite of existing models
        saved_path = h2o.save_model(model=model, path=path, force=True)
        
        # Confirm successful save with the full path
        print(f"Model saved successfully to: {saved_path}")
        return saved_path
    
    except Exception as e:
        # Catch and report any errors during the model saving process
        print(f"Error saving the model: {e}")
        return None
    
def get_model_evaluation(model, df):
    """
    Generates a comprehensive performance evaluation report for a given model.
    
    Parameters:
    - model: H2O model to evaluate
    - df: H2O DataFrame for performance testing
    
    Returns:
    str: Markdown-formatted performance metrics report
    
    Evaluation Strategy:
    - Dynamically selects metrics based on model type
    - Supports both classification and regression models
    - Provides detailed, human-readable performance insights
    """
    # Get model performance metrics on the test dataset
    perf = model.model_performance(test_data=df)
    
    # Determine the model's category (classification or regression)
    model_params = model._model_json['output']['model_category']
    
    report = ""

    # Dynamically generate performance report based on model type
    if model_params in ["Binomial", "Multinomial"]:
        # Metrics for classification models
        report += "### Model Performance Metrics:\n"
        report += f"- Model Type: Classification Model\n"
        report += f"- **AUC**: {perf.auc()}\n"
        report += f"- **Accuracy**: {perf.accuracy()}\n"
        report += f"- **LogLoss**: {perf.logloss()}\n"
        report += f"- **Mean Per Class Error**: {perf.mean_per_class_error()}\n"
        report += f"- **Precision-Recall AUC (AUC PR)**: {perf.aucpr()}\n"

    elif model_params == "Regression":
        # Metrics for regression models
        report += "### Model Performance Metrics:\n"
        report += f"- Model Type: Regression Model\n"
        report += f"- **RMSE**: {perf.rmse()}\n"
        report += f"- **MSE**: {perf.mse()}\n"
        report += f"- **R2**: {perf.r2()}\n"

    else:
        # Fallback for unrecognized model types
        report += "## Unable to determine model type automatically.\n"

    return report