import pandas as pd
import numpy as np
import logging
import os
from scipy.stats.mstats import winsorize

# Configure logging for better tracking of events
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(filepath):
    """
    Loads the weather dataset from a CSV file and performs basic validations.
    
    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame if successful, otherwise None.
    """
    try:
        # Check if file exists before attempting to load
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # Ensure the file is not empty
        if df.empty:
            raise ValueError("Loaded file is empty.")

        logging.info(f" Data loaded successfully from {filepath}")
        logging.info(f" Dataset shape: {df.shape}")
        logging.info(f" Missing values before handling:\n{df.isnull().sum()}")

        return df

    except FileNotFoundError as e:
        logging.error(e)
    except pd.errors.EmptyDataError:
        logging.error(" Error: The CSV file is empty.")
    except pd.errors.ParserError:
        logging.error(" Error: CSV file is not properly formatted.")
    except Exception as e:
        logging.error(f" Unexpected error: {e}")

    return None


def handle_missing_values(df):
    """
    Handles missing values by filling numerical columns with the median and categorical columns with the mode.
    
    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    missing_before = df.isnull().sum().sum()

    # Fill missing values in numerical columns with the median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()), axis=0)

    # Fill missing values in categorical columns with the mode (most frequent value)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0] if not col.mode().empty else "Unknown"), axis=0)

    missing_after = df.isnull().sum().sum()

    logging.info(f" Missing values handled. Before: {missing_before}, After: {missing_after}")
    return df


def remove_outliers(df, skew_threshold=1.5):
    """
    Detects and handles outliers using Winsorization and log transformation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        skew_threshold (float): Threshold for applying log transformation on skewed features.

    Returns:
        pd.DataFrame: DataFrame with outliers handled.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        # Compute Interquartile Range (IQR)
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        # Detect outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_percentage = len(outliers) / len(df)

        # Compute skewness
        skewness = df[col].skew()

        if outlier_percentage > 0.1:
            # If more than 10% of data are outliers, apply Winsorization
            logging.warning(f" High outlier percentage in {col} ({outlier_percentage:.2%}), applying Winsorization.")
            df[col] = winsorize(df[col], limits=[0.05, 0.05])

        elif abs(skewness) > skew_threshold:
            # If feature is highly skewed, apply log transformation
            logging.info(f" Applying log transformation to {col} (Skewness: {skewness:.2f})")
            df[col] = np.log1p(df[col])  # log1p handles zero values safely

        # Feature Engineering: Add an outlier flag column
        df[f"{col}_outlier_flag"] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

    logging.info(" Outlier handling completed.")
    return df


def detect_duplicates(df):
    """
    Detects and removes duplicate rows from the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without duplicate records.
    """
    before_dedup = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_dedup = df.shape[0]

    logging.info(f" Removed {before_dedup - after_dedup} duplicate rows.")
    return df


def preprocess_data(filepath, output_filepath):
    """
    Executes the full data preprocessing pipeline: 
    - Loads data
    - Handles missing values
    - Removes outliers
    - Detects duplicates
    - Saves cleaned data to a new file

    Args:
        filepath (str): Path to the raw CSV file.
        output_filepath (str): Path to save the cleaned dataset.
    """
    df = load_data(filepath)
    if df is None:
        return

    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = detect_duplicates(df)

    try:
        df.to_csv(output_filepath, index=False)
        logging.info(f" Data cleaning completed. Cleaned dataset saved to: {output_filepath}")
    except Exception as e:
        logging.error(f" Error saving file: {e}")


# Run preprocessing when script is executed directly
if __name__ == "__main__":
    input_filepath = "data/GlobalWeatherRepository.csv"
    output_filepath = "data/cleaned_weather_data.csv"
    
    preprocess_data(input_filepath, output_filepath)
