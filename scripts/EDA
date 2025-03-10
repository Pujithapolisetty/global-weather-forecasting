import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
EDA_RESULTS_DIR = "EDA_results"
FIGURES_DIR = os.path.join(EDA_RESULTS_DIR, "figures")

# Create necessary directories
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f" Data loaded successfully from {filepath}.")
    print(f" Dataset Shape: {df.shape}")
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values by:
    - Filling numeric columns with their median.
    - Filling categorical columns with their most frequent value (mode).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Handle numeric columns
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Handle categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(mode_value, inplace=True)

    print(" Missing values handled.")
    return df

def remove_outliers(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Removes outliers using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Multiplier for IQR (default is 1.5).

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    original_size = len(df)

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - (threshold * IQR), Q3 + (threshold * IQR)

        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    removed_rows = original_size - len(df)
    print(f" Removed {removed_rows} outlier rows.")
    return df

def save_summary_statistics(df: pd.DataFrame) -> None:
    """
    Saves dataset summary statistics as a CSV file.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    summary_path = os.path.join(EDA_RESULTS_DIR, "summary_statistics.csv")
    df.describe().to_csv(summary_path)
    print(f" Summary statistics saved to {summary_path}")

def save_missing_values_report(df: pd.DataFrame) -> None:
    """
    Saves missing value counts as a CSV file.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    missing_path = os.path.join(EDA_RESULTS_DIR, "missing_values.csv")
    df.isnull().sum().to_csv(missing_path)
    print(f" Missing values report saved to {missing_path}")

def save_pairplot(df: pd.DataFrame) -> None:
    """
    Generates and saves a pairplot of numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        sns.pairplot(numeric_df)
        fig_path = os.path.join(FIGURES_DIR, "pairplot.png")
        plt.savefig(fig_path)
        plt.close()
        print(f" Pairplot saved to {fig_path}")

def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Generates and saves a correlation heatmap of numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")

        heatmap_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f" Correlation heatmap saved to {heatmap_path}")

def preprocess_and_analyze(filepath: str) -> None:
    """
    Executes the full data preprocessing and Exploratory Data Analysis (EDA) pipeline.

    Args:
        filepath (str): Path to the CSV file.
    """
    print("\n Starting Data Preprocessing & EDA...")

    df = load_data(filepath)
    df = handle_missing_values(df)
    df = remove_outliers(df)

    # Save EDA reports
    save_summary_statistics(df)
    save_missing_values_report(df)
    save_pairplot(df)
    save_correlation_heatmap(df)

    print("\n EDA process completed successfully!")

if __name__ == "__main__":
    preprocess_and_analyze("data/GlobalWeatherRepository.csv")
