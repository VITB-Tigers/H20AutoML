# METADATA [app.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Description: This code script contains the Streamlit web app for training AutoML models using H2O AutoML.
# Developed By: 
    # Name: Vansh Raja
    # Role: Intern, PreProd Corp
    # Code ownership rights: Vansh Raja, PreProd Corp

# Version: 1.0 [Date: 12-12-2024]
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dependencies:
    # Python 3.11.15
    # Libraries:
        # Streamlit 1.40.2
        # H2O==3.46.0.6
        
# Importing the required libraries
import streamlit as st  # Web application framework for creating interactive web apps
import h2o  # Machine learning library for automated machine learning tasks

# Importing the required functions from custom modules
from automl import process_data, trainAutoML, get_model_ids, save_model, get_model_evaluation  # AutoML related functions
from db_utils import connect_to_redis, write_to_redis, load_from_redis, drop_db, get_column_names  # Database utility functions
from transform import (
    convert_column_to_categorical,  # Convert columns to categorical type
    convert_column_to_numeric,      # Convert columns to numeric type
    scale_selected_columns,          # Scale numeric columns
    impute_missing_values,           # Handle missing values
    drop_columns,                    # Remove specific columns
    log_transform_columns,           # Apply log transformation to columns
    sqrt_transform_columns,           # Apply square root transformation to columns
    split_dataset                    # Split dataset into training and testing data
)

# Set the page configuration for the Streamlit web application
# Configures the page title, icon, and layout
st.set_page_config(
    page_title="AutoML",           # Title displayed in browser tab
    page_icon=":bar_chart:",       # Emoji/icon for the page
    layout="centered"               # Center-aligned layout
)

# Set the logo for the web application
# Uses an external image URL and provides a link when logo is clicked
st.logo(
    image="https://i.imgur.com/zXj45M2.png",        # Logo image URL
    link="https://thepreprodcorp.com/"              # Website link when logo is clicked
)

# Create a centered heading for the web application using markdown
# Allows custom styling of the title
st.markdown("<h1 style='text-align: center; color: white;'>AutoML </h1>", unsafe_allow_html=True)

# Add a divider to separate the heading from the content
st.divider()

# Create tabs for different sections of the web application
# This allows for a clean, organized interface with multiple functional areas
tab1, tab2, tab3, tab4 = st.tabs([
    "Configuration",            # Tab for setting up configurations
    "Data Ingestion",           # Tab for uploading and loading data
    "Data Transformation",      # Tab for preprocessing and transforming data
    "Auto Train ML Models"      # Tab for automated machine learning model training
])

# Initialize Streamlit session state variables
# Session state allows preserving variables across reruns of the Streamlit app

# Flag to ensure H2O is initialized only once
if "h2o_initialized" not in st.session_state:
    st.session_state.h2o_initialized = False
    
# Flag to track if models have been trained
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    
# List to store trained model IDs
if "model_ids" not in st.session_state:
    st.session_state.model_ids = []
    
# Variable to store model leaderboard (performance comparison)
if "model_leaderboard" not in st.session_state:
    st.session_state.model_leaderboard = None

# Variables for Redis configuration
if "host" not in st.session_state:
    st.session_state.host = "localhost"

if "port" not in st.session_state:
    st.session_state.port = 6379

if "db" not in st.session_state:
    st.session_state.db = 1
    
# Initialize H2O machine learning framework
# Checks if H2O has already been initialized to avoid repeated initialization
if not st.session_state.h2o_initialized:
    h2o.init()  # Start H2O cluster
    st.session_state.h2o_initialized = True  # Mark H2O as initialized

# Tab for Configuration
with tab1:
    # Subheader for the configuration section
    st.subheader("Configuration")
    
    # Create a form for Redis configuration
    # Using a form ensures all inputs are collected before submission
    with st.form(key="Configure Redis"):
        # Subheader for Redis configuration section
        st.subheader("Redis Configuration")
        
        # Input field for Redis host
        # Default value is localhost, with a help tooltip
        host = st.text_input(
            "Host", 
            value=st.session_state.host, 
            help="Enter the hostname of the Redis server"
        )
        
        # Input field for Redis port
        # Default port is 6379 (standard Redis port), with a help tooltip
        port = st.number_input(
            "Port", 
            value=st.session_state.port, 
            help="Enter the port number of the Redis server"
        )
        
        # Input field for Redis database number
        # Default database is 1, with a help tooltip
        db = st.number_input(
            "Database", 
            value=st.session_state.db, 
            help="Enter the database number to connect to"
        )
        
        # Form submission button to check Redis connection
        if st.form_submit_button("Check redis connection", use_container_width=True):
            # Attempt to connect to Redis using provided configuration
            r = connect_to_redis(host, port, db)
            
            # Display success message if connection is established
            st.success("Connected to Redis successfully!", icon="✅")

# Tab for Data Ingestion section in a Streamlit application
with tab2:
    # Create a subheader for the Data Ingestion section
    st.subheader("Data Ingestion")
    
    # Create a container with a border for data upload options
    with st.container(border=True):
        # Radio button to choose between entering file path or uploading file
        # Horizontal layout for better user experience
        dataset_choice = st.radio("Choose an option to upload the data", ["Enter the path", "Upload the file"], horizontal=True)
        
        # Determine whether path input is enabled based on user's choice
        # If "Enter the path" is selected, dataset_bool will be True
        dataset_bool = True if dataset_choice == "Enter the path" else False
        
        # Text input for file location
        # Disabled if file upload option is selected
        file_location = st.text_input("Path of the file",
                                     placeholder="Dataset Path", 
                                     help="Enter the complete path to the source data",
                                     disabled=not dataset_bool)
        
        # Text input for file name
        # Disabled if file upload option is selected
        file_name = st.text_input("Name of the file",
                                     placeholder="Dataset Name",
                                     help="Enter the complete name with extension, i.e., .csv or .xlsx", 
                                     disabled=not dataset_bool)

        # Create a centered "OR" separator between path input and file upload
        st.markdown("<h4 style='text-align: center;'>OR</h4>", unsafe_allow_html=True)

        # File uploader widget
        # Accepts only CSV and Excel files
        # Disabled if path input option is selected
        dataset_upload = st.file_uploader("Upload the file",
                                     type=["csv", "xlsx"],
                                     help="Upload the file to ingest the data",
                                     disabled=dataset_bool)
            
        # Ingest button to process and store the data
        if st.button("Ingest", use_container_width=True):
            # If path input method is selected
            if dataset_bool:
                # Remove trailing slash from file location if present
                if file_location[-1] == "/":
                    file_location = file_location[:-1]
                
                # Construct full file path
                file_path =f"{file_location}/{file_name}"
                # Process data from the specified file path
                data = process_data(file_path)
            else:
                # Process data from uploaded file
                data = process_data(dataset_upload)
            
            # Connect to Redis database
            r = connect_to_redis(host, port, db)
            
            # Data ingestion and storage logic
            if data is not None:
                # Attempt to write data to Redis
                if write_to_redis("Data", data, r) is not None:
                    # Success message if data is stored successfully
                    st.success("Data ingested successfully!", icon="✅")
                else:
                    # Error message if writing to Redis fails
                    st.error("Error ingesting data!", icon="❌")
            else:
                # Error message if data processing fails
                st.error("Error uploading data!", icon="❌")
    
    # Create a form for visualizing data dimensions
    with st.form(key="Visualise Data"):
        # Subheader with help text explaining data dimensions
        st.subheader("Data dimensions", help="Displays the total number of columns in the dataset")

        # Reconnect to Redis to load the ingested data
        r = connect_to_redis(host, port, db)
        # Load data from Redis
        data = load_from_redis("Data", r)
        # Create a container to display data information
        containter = st.container()

        # Form submit button to display data dimensions
        if st.form_submit_button("Run", use_container_width=True):
            # Check if data exists in Redis
            if data is None:
                # Error message if no data is found
                st.error("No data found in the database!", icon="❌")
            else:
                # Extract and display number of rows and columns
                rows, columns = data.shape
                containter.write(f"Number of rows: {rows}")
                containter.write(f"Number of columns: {columns}")
                # Display first few rows of the dataset
                containter.write(data.head(), hide_index=True)
            
    # Container for database reset functionality
    with st.container(border=True):
        # Subheader for database reset section
        st.subheader("Reset Database")
        
        # Button to drop the entire database
        if st.button("Drop Database", help="This will delete the database and all the data stored in it.", use_container_width=True):
            # Attempt to drop the database
            if drop_db(r):
                # Remove all H2O models
                h2o.remove_all()
                # Reset model training state
                st.sesssion_state.model_trained = False
                # Reconnect to Redis after dropping the database
                r = connect_to_redis(host, port, db)
                # Success message
                st.success("Database dropped successfully!", icon="✅")
            else:
                # Error message if database drop fails
                st.error("Error dropping database!", icon="❌")

# Tab for Data Transformation
with tab3:
    # Create a subheader for the Data Transformation section
    st.subheader("Data Transformation")
    
    # Connect to Redis database
    r = connect_to_redis(host, port, db)
    
    # Check if data exists in the database
    if get_column_names("Data", r) is None:
        # If no data is found, initialize an empty list of columns
        current_rows = []
        # Display a warning message to ingest data first
        st.warning("No data found in the database!, Please ingest data first.")
    else:
        # If data exists, get the list of column names
        current_rows = list(get_column_names("Data", r))

    # Expandable section about transformation operations
    with st.expander("Learn more about the transformation operations available here:"):
        transformations_description = """
        Target Variable Selection:
        - Identifies the column to be predicted in machine learning models, crucial for supervised learning tasks.

        Feature Removal:
        - Allows removing unnecessary or redundant columns to simplify the dataset and improve model performance.

        Numerical Conversion:
        - Transforms columns to numeric type, ensuring proper data type for mathematical operations and model training.

        Categorical Conversion:
        - Converts columns to categorical type, particularly useful for tree-based algorithms that handle categorical features differently.

        Numeric Column Scaling:
        - Normalizes numeric features to a standard scale, preventing features with larger magnitudes from dominating model training.

        Missing Value Imputation:
        - Fills in missing data using statistical methods like mean, median, or mode to prepare the dataset for analysis.

        Log Transformation:
        - Applies logarithmic scaling to reduce skewness and handle exponentially distributed data.

        Square Root Transformation:
        - Helps in stabilizing variance and normalizing data with right-skewed distributions.
        """
        st.markdown(transformations_description)
    
    
    # Form to set the target variable for machine learning
    with st.form(key="Set Target Variable"):
        # Dropdown to select the target variable from existing columns
        target_variable = st.selectbox(
            "Target Variable",
            options=current_rows,
            placeholder="Target Variable",
            help="Enter the name of the target variable"
        )
        
        # Form submission button to set the target variable
        if st.form_submit_button("Set Target Variable", use_container_width=True):
            # Display success message when target variable is set
            st.success("Target variable set successfully!", icon="✅")   
    
    # Form to remove selected features from the dataset
    with st.form(key="Config Remove"):
        # Multiselect to choose features to remove
        removal_selection = st.multiselect(
            "Remove features",
            options=current_rows,
            help="Enter the names of features to remove"
        )
        
        # Form submission button to remove selected features
        if st.form_submit_button(label="Remove Features", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Remove selected columns
            data = drop_columns(data, removal_selection)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Features removed successfully!", icon="✅")    
    
    # Form to convert selected columns to numeric type
    with st.form(key="Convert Numerical"):
        # Multiselect to choose columns to convert to numeric
        number_selection = st.multiselect(
            "Convert to numbers",
            options=current_rows, 
            help="Enter the names of features to convert to numbers"
        )
        
        # Form submission button to convert selected columns
        if st.form_submit_button("Convert Feature(s)", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Convert selected columns to numeric
            data = convert_column_to_numeric(data, number_selection)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Feature(s) converted successfully!", icon="✅")
            
    # Form to convert selected columns to categorical type
    with st.form(key="Convert Categorical"):
        # Multiselect to choose columns to convert to categorical
        categorical_selection = st.multiselect(
            "Convert to categorical",
            options=current_rows,
            help="Enter the names of features to convert to categorical. This is especially useful for algorithms like trees with numerical features that should be treated as categorical."
        )
        
        # Form submission button to convert selected columns
        if st.form_submit_button("Convert to Categorical", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Convert selected columns to categorical
            data = convert_column_to_categorical(data, categorical_selection)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Feature(s) converted successfully!", icon="✅")
            
    # Form to scale selected numeric columns
    with st.form(key="Scale Numeric Columns"):
        # Multiselect to choose numeric columns to scale
        scaling_selection = st.multiselect(
            "Scale numeric columns",
            options=current_rows,
            help="Enter the names of numeric features to scale"
        )
        
        # Form submission button to scale selected columns
        if st.form_submit_button("Scale Numeric Columns", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Scale selected numeric columns
            data = scale_selected_columns(data, scaling_selection)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Numeric columns scaled successfully!", icon="✅")
    
    # Form to impute missing values
    with st.form(key="Impute Missing Values"):
        # Multiselect to choose columns with missing values to impute
        impute_selection = st.multiselect(
            "Impute missing values",
            options=current_rows,
            help="Enter the names of features to impute missing values"
        )
        
        # Dropdown to select imputation method
        impute_method = st.selectbox(
            "Imputation method",
            options=["mean", "median", "mode"],
            help="Select the method to impute missing values"
        )
        
        # Form submission button to impute missing values
        if st.form_submit_button("Impute Missing Values", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Impute missing values using selected method
            data = impute_missing_values(data, impute_selection, impute_method)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Missing values imputed successfully!", icon="✅")
            
    # Form to apply log transformation to selected columns
    with st.form(key="Log Transform"):
        # Multiselect to choose columns for log transformation
        log_transform_selection = st.multiselect(
            "Apply log transformation",
            options=current_rows,
            help="Enter the names of features to apply log transformation"
        )
        
        # Form submission button to apply log transformation
        if st.form_submit_button("Apply Log Transformation", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Apply log transformation to selected columns
            data = log_transform_columns(data, log_transform_selection)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Log transformation applied successfully!", icon="✅")
            
    # Form to apply square root transformation to selected columns
    with st.form(key="Square Root Transform"):
        # Multiselect to choose columns for square root transformation
        sqrt_transform_selection = st.multiselect(
            "Apply square root transformation",
            options=current_rows,
            help="Enter the names of features to apply square root transformation"
        )
        
        # Form submission button to apply square root transformation
        if st.form_submit_button("Apply Square Root Transformation", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            # Apply square root transformation to selected columns
            data = sqrt_transform_columns(data, sqrt_transform_selection)
            # Save the modified data back to Redis
            write_to_redis("Data", data, r)
            # Display success message
            st.success("Square root transformation applied successfully!", icon="✅") 
    
    # Form to visualize data dimensions
    with st.form(key="Visualise Data 2"):
        # Subheader for data dimensions section
        st.subheader("Data dimensions", help="Displays the total number of columns in the dataset")

        # Reconnect to Redis and load data
        r = connect_to_redis(host, port, db)
        data = load_from_redis("Data", r)
        # Create a container to display data information
        containter = st.container()

        # Form submission button to display data dimensions
        if st.form_submit_button("Run", use_container_width=True):
            # Check if data exists
            if data is None:
                # Display error if no data is found
                st.error("No data found in the database!", icon="❌")
            else:
                # Get and display number of rows and columns
                rows, columns = data.shape
                containter.write(f"Number of rows: {rows}")
                containter.write(f"Number of columns: {columns}")
                # Display first few rows of the dataset
                containter.write(data.head(), hide_index=True)

# Tab for Auto Training ML Models
with tab4:
    # Container for Train-Test Split section with a border
    with st.container(border=True):
        # Markdown header for train-test split
        st.markdown("### Train Test Split Data")
        
        # Create two columns for input layout
        col1, col2 = st.columns(2)
        
        # First column: Training data size input
        with col1:
            # Number input for training data percentage
            train_size = st.number_input(
                "Training Data Split",
                placeholder="% of training data",
                help="Enter the percentage of data to be used for training",
                value=70  # Default 70% training data
            )
        
        # Calculate test size automatically
        test_size = 100 - train_size
        
        # Second column: Test data size display (disabled)
        with col2:
            # Display test size, disabled to prevent manual editing
            test_size = st.number_input(
                "Testing Data Split",
                placeholder="% of testing data",
                help="See the percentage of data to be used for testing",
                value=test_size, 
                disabled=True  # Read-only field
            )
        
        # Button to perform data splitting
        if st.button("Split Data", use_container_width=True):
            # Load data from Redis
            data = load_from_redis("Data", r)
            
            # Split dataset using specified train percentage
            train, test = split_dataset(data, [train_size/100])
            
            # Save train and test datasets to Redis
            write_to_redis("Train Data", train, r)
            write_to_redis("Test Data", test, r)
            
            # Display success message
            st.success("Data split successfully!", icon="✅")
            
    # Divider for visual separation
    st.divider()
    
    # Header for ML Model Training section
    st.header("Train ML Models")
    
    # Form for Automated ML Model Training
    with st.form(key="Auto Train Best ML Models"):
        # List of available model types
        model_names = ["DRF", "GLM", "GBM", "XGBoost", "StackedEnsemble", "DeepLearning"]
        
        # Multiselect for model types
        model_options = st.multiselect(
            label="Select models to train", 
            options=model_names, 
            help="Select the models to train, or leave empty to train all models", 
            placeholder="Select models to train, or leave empty to train all models"
        )
        
        # Expandable section with model descriptions
        with st.expander("View Model Descriptions"):
            # Descriptions of each model type
            descriptions = """ 
            DRF: Distributed Random Forest - Best for handling high-dimensional datasets with complex interactions and when you need robust feature importance analysis.
            
            GLM: Generalized Linear Model - Ideal for linear relationships, interpretable models, and when working with statistical inference or clearly defined linear problems.
            
            GBM: Gradient Boosting Machine - Excellent for achieving high predictive performance, handling mixed data types, and when sequential error correction is crucial.
            
            XGBoost: Extreme Gradient Boosting - Perfect for competition-level predictive modeling, handling sparse data, and when computational efficiency is a priority.
            
            StackedEnsemble: Stacked Ensemble - Optimal when you want to combine strengths of multiple models, reducing bias and improving overall prediction accuracy.
            
            DeepLearning: Deep Learning Models - Best suited for complex, non-linear problems with large amounts of data, especially in image, text, and time series analysis.
            """
            
            # Display descriptions as markdown
            st.markdown(descriptions)
        
        # If no models selected, set to None (train all)
        if model_options == []:
            model_options = None
        
        # Create 4 columns for additional training parameters
        col1, col2, col3, col4 = st.columns(4)
        
        # Maximum runtime for entire AutoML process
        with col1:
            max_runtime_secs = st.number_input(
                "Max Runtime (seconds)", 
                min_value=0, 
                value=60, 
                help="Maximum time in seconds for the AutoML to run", 
                step=5
            )
        
        # Maximum runtime per individual model
        with col2:
            max_runtime_secs_per_model = st.number_input(
                "Max Runtime Per Model", 
                min_value=0, 
                value=0, 
                help="Maximum time in seconds for each model to run", 
                step=5
            )
            
        # Maximum number of models to train
        with col3:
            max_models = st.number_input(
                "Max Models", 
                min_value=0, 
                value=5, 
                help="Maximum number of models to train"
            )
            
        # Random seed for reproducibility
        with col4:
            seed = st.number_input(
                "Seed", 
                min_value=0, 
                value=1, 
                help="Seed for reproducibility"
            )
        
        # Project name input
        project_name = st.text_input(
            "Project Name", 
            help="Enter a name for the project, this will help train more models in the same leaderboard", 
            value="AutoML"
        )
        
        # Model training submission button
        if st.form_submit_button("Train Models", use_container_width=True):
            # Show spinner during model training
            with st.spinner("Training models..."):
                # Load training data from Redis
                train_data = load_from_redis("Train Data", r)
                
                # Train AutoML models
                model_leaderboard, model_leader = trainAutoML(
                    train_data,
                    target_variable, 
                    project_name, 
                    model_options, 
                    max_runtime_secs=max_runtime_secs, 
                    max_models=max_models, 
                    max_runtime_secs_per_model=max_runtime_secs_per_model, 
                    seed=seed
                )
                
                # Update session state with training results
                st.session_state.model_trained = True
                st.session_state.model_ids = get_model_ids(model_leaderboard)
                st.session_state.model_leaderboard = model_leaderboard.as_data_frame()
                
                # Display success message
                st.success("Model trained successfully!", icon="✅")
            
            # Display model leaderboard
            st.write(model_leaderboard)
    
    # Container for Model Evaluation
    with st.container(key="Model Evaluation", border=True):
        # Subheader for model evaluation section
        st.subheader("Model Evaluation")
        
        # Check if models have been trained
        if st.session_state.model_trained:
            # Dropdown to select a trained model
            model_id = st.selectbox(
                "Select a model to see its evaluation", 
                st.session_state.model_ids, 
                help="Select a model to evaluate"
            )    
        
            # Load test data from Redis
            test_data = load_from_redis("Test Data", r)
            
            # Retrieve the selected model
            model = h2o.get_model(model_id)
            
            # Get model evaluation metrics
            evaluation = get_model_evaluation(model, test_data)
            
            # Display evaluation metrics
            st.write(evaluation)
            
            # Expandable section for model leaderboard
            with st.expander("View the current Model Leaderboard"):
                st.write(st.session_state.model_leaderboard)
            
            # Input for model save path
            model_save_path = st.text_input(
                "Model Save Path", 
                help="Enter the path to save the model",
                value="Models/"
            )
            
            # Button to save current model
            if st.button("Save Current Model", help="Save the current model"):
                save_model(model_id, model_save_path)
                st.success(f"Model {model_id} saved successfully!", icon="✅")
                
        else:
            # Warning if no models have been trained
            st.warning("Train a model to evaluate it!")  

# END OF SCRIPT ---------------------------------------------------------------------------------------------