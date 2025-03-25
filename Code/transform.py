# METADATA [transform.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Description: This file contains the functions to carry out all data transformations using H2O.
# Developed By: 
    # Name: Vansh Raja

# Version: 1.0 [Date: 12-12-2024]
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dependencies:
    # Python 3.11.15
    # Libraries:
        # H2O 3.46.0.6


from h2o.transforms.preprocessing import H2OScaler

def convert_column_to_categorical(df, column_name):
    """
    Converts a specified column in an H2OFrame to a categorical data type.
    
    Purpose:
    - Transform numeric or string columns to categorical for machine learning
    - Prepare data for algorithms that require categorical input
    
    Parameters:
    - df: H2OFrame containing the target column
    - column_name (str): Name of the column to convert
    
    Workflow:
    1. Use H2O's asfactor() method to convert column type
    2. Provide feedback on successful conversion
    
    Returns:
    - Modified H2OFrame with the specified column converted to categorical
    - None if conversion fails
    
    Use Cases:
    - Preparing target variables for classification
    - Encoding non-numeric features
    - Preparing data for categorical machine learning algorithms
    """
    try:
        # Convert column to categorical using H2O's asfactor() method
        df[column_name] = df[column_name].asfactor()
        print(f"Column '{column_name}' successfully converted to categorical.")
        return df
    except Exception as e:
        # Comprehensive error handling and logging
        print(f"Error converting column to categorical: {e}")
        return None

def convert_column_to_numeric(df, column_names):
    """
    Converts specified columns in an H2OFrame to numeric data type.
    
    Purpose:
    - Transform categorical or string columns to numeric
    - Prepare data for numerical machine learning algorithms
    
    Parameters:
    - df: H2OFrame containing columns to convert
    - column_names (list): List of column names to convert to numeric
    
    Workflow:
    1. Iterate through specified columns
    2. Use H2O's asnumeric() method for conversion
    3. Provide feedback for each column conversion
    
    Returns:
    - Modified H2OFrame with specified columns converted to numeric
    - None if conversion fails
    
    Use Cases:
    - Preparing encoded categorical variables for numeric analysis
    - Standardizing column types for machine learning models
    - Converting string representations of numbers
    """
    try:
        # Convert each specified column to numeric
        for col in column_names:
            df[col] = df[col].asnumeric()
            print(f"Column '{col}' successfully converted to numeric.")
        return df
    except Exception as e:
        # Comprehensive error handling and logging
        print(f"Error converting columns to numeric: {e}")
        return None

def scale_selected_columns(df, columns_to_scale):
    """
    Applies standardization scaling to selected numeric columns using H2OScaler.
    
    Purpose:
    - Normalize selected features to have zero mean and unit variance
    - Prepare data for machine learning algorithms sensitive to scale
    
    Parameters:
    - df: H2OFrame to be scaled
    - columns_to_scale (list): List of column names to scale
    
    Scaling Workflow:
    1. Create H2OScaler instance
    2. Extract specified columns
    3. Fit and transform the selected columns
    4. Replace original columns with scaled versions
    
    Returns:
    - H2OFrame with specified columns scaled
    - None if scaling fails
    
    Key Characteristics:
    - Uses H2O's built-in scaling mechanism
    - Preserves original DataFrame structure
    - Applies scaling only to selected columns
    
    Recommended For:
    - Algorithms sensitive to feature scaling (e.g., SVM, Neural Networks)
    - Comparisons across features with different scales
    - Improving model convergence
    """
    try:
        # Initialize H2OScaler for standardization
        scaler = H2OScaler()

        # Extract specified columns as a new H2OFrame
        selected_columns = df[columns_to_scale]

        # Fit and transform the selected columns
        scaler.fit(selected_columns)
        scaled_columns = scaler.transform(selected_columns)

        # Replace the scaled columns back into the original H2OFrame
        df[columns_to_scale] = scaled_columns
        
        print("Selected columns successfully scaled using H2OScaler.")
        return df
    except Exception as e:
        # Detailed error logging
        print(f"Error scaling selected columns: {e}")
        return None

def split_dataset(df, ratios=[0.7], seed=1234):
    """
    Splits an H2OFrame into training and test sets with reproducibility.
    
    Purpose:
    - Divide dataset into training and testing subsets
    - Ensure consistent, reproducible splits for machine learning workflows
    
    Parameters:
    - df: H2OFrame to be split
    - ratios (list): Proportions for splitting the dataset (default: 70% train, 30% test)
    - seed (int): Random seed for reproducible splitting
    
    Splitting Strategy:
    - Uses H2O's built-in split_frame method
    - Supports custom split ratios
    - Ensures reproducibility through seed
    
    Returns:
    - train: Training subset of the original DataFrame
    - test: Testing subset of the original DataFrame
    - None if splitting fails
    
    Best Practices:
    - Always set a consistent seed for reproducibility
    - Adjust ratios based on dataset size and problem complexity
    """
    try:
        # Print the splitting ratios for transparency
        print(ratios)
        
        # Split the DataFrame using specified ratios and seed
        train, test = df.split_frame(ratios=ratios, seed=seed)
        
        print("Dataset successfully split into train and test sets.")
        return train, test
    except Exception as e:
        # Comprehensive error handling
        print(f"Error splitting dataset: {e}")
        return None

def impute_missing_values(df, column_names, method='mean'):
    """
    Impute missing values in specified columns of an H2OFrame.
    
    Purpose:
    - Handle missing data in machine learning datasets
    - Provide flexible imputation strategies
    
    Parameters:
    - df: H2OFrame containing columns with missing values
    - column_names (list): Columns to impute
    - method (str): Imputation method ('mean', 'median', 'mode')
    
    Imputation Workflow:
    1. Iterate through specified columns
    2. Apply chosen imputation method
    3. Use interpolation for smooth value replacement
    
    Returns:
    - H2OFrame with missing values imputed
    - None if imputation fails
    
    Imputation Strategies:
    - Mean: Replace with column average (good for normally distributed data)
    - Median: Replace with middle value (robust to outliers)
    - Mode: Replace with most frequent value (best for categorical data)
    """
    try:
        # Impute each specified column using the chosen method
        for column_name in column_names:
            df.impute(column=column_name, method=method, combine_method='interpolate')
            print(f"Missing values in column '{column_name}' successfully imputed using method '{method}'.")
        return df
    except Exception as e:
        # Detailed error logging
        print(f"Error imputing missing values in columns: {e}")
        return None

def drop_columns(df, columns):
    """
    Removes specified columns from an H2OFrame.
    
    Purpose:
    - Eliminate unnecessary or redundant columns
    - Streamline dataset for machine learning
    
    Parameters:
    - df: H2OFrame to modify
    - columns (list): Columns to remove
    
    Dropping Strategy:
    - Uses H2O's built-in drop method
    - Removes multiple columns in a single operation
    
    Returns:
    - H2OFrame with specified columns removed
    - None if column dropping fails
    
    Use Cases:
    - Removing highly correlated features
    - Eliminating irrelevant columns
    - Reducing dimensionality
    """
    try:
        # Drop specified columns from the DataFrame
        df = df.drop(columns)
        print("Columns successfully dropped.")
        return df
    except Exception as e:
        # Comprehensive error handling
        print(f"Error dropping columns: {e}")
        return None

def log_transform_columns(df, column_names):
    """
    Applies logarithmic transformation to specified columns.
    
    Purpose:
    - Reduce skewness in highly skewed distributions
    - Compress the scale of extreme values
    - Stabilize variance for certain statistical analyses
    
    Parameters:
    - df: H2OFrame containing columns to transform
    - column_names (list): Columns to apply log transformation
    
    Transformation Workflow:
    1. Iterate through specified columns
    2. Apply logarithmic transformation
    3. Provide feedback on successful transformation
    
    Returns:
    - H2OFrame with log-transformed columns
    - None if transformation fails
    
    Recommended For:
    - Positively skewed numerical features
    - Features with exponential growth patterns
    - Improving normality of data distribution
    
    Caution:
    - Only applicable to positive values
    - May not be suitable for zero or negative values
    """
    try:
        # Apply log transformation to each specified column
        for column_name in column_names:
            df[column_name] = df[column_name].log()
            print(f"Log transformation successfully applied to column '{column_name}'.")
        return df
    except Exception as e:
        # Detailed error logging
        print(f"Error applying log transformation: {e}")
        return None

def sqrt_transform_columns(df, column_names):
    """
    Applies square root transformation to specified columns.
    
    Purpose:
    - Moderate the effect of large values
    - Reduce skewness in moderately skewed distributions
    - Stabilize variance for certain analyses
    
    Parameters:
    - df: H2OFrame containing columns to transform
    - column_names (list): Columns to apply square root transformation
    
    Transformation Workflow:
    1. Iterate through specified columns
    2. Apply square root transformation
    3. Provide feedback on successful transformation
    
    Returns:
    - H2OFrame with square root-transformed columns
    - None if transformation fails
    
    Recommended For:
    - Count data
    - Slightly right-skewed distributions
    - Features with non-linear relationships
    
    Characteristics:
    - Less aggressive than log transformation
    - Preserves more of the original data's characteristics
    - Suitable for non-negative values
    """
    try:
        # Apply square root transformation to each specified column
        for column_name in column_names:
            df[column_name] = df[column_name].sqrt()
            print(f"Square root transformation successfully applied to column '{column_name}'.")
        return df
    except Exception as e:
        # Comprehensive error handling
        print(f"Error applying square root transformation: {e}")
        return None