# data_utils.py
# Functions for fetching stock data and calculating technical indicators.

import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna

# Import necessary configurations
import config # Assuming config.py is in the same directory

def fetch_stock_data(ticker, start, end):
    """
    Fetches historical stock data for a given ticker and date range
    and calculates a comprehensive set of technical indicators.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start (str): The start date in 'YYYY-MM-DD' format.
        end (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with numeric features (OHLCV + indicators).
            - pd.Series: Series containing the target variable (shifted close price).

    Raises:
        ValueError: If no data is fetched for the ticker or required columns are missing.
    """
    print(f"Fetching data for {ticker} from {start} to {end}...")
    # Use interval='1d' explicitly, although it's often the default
    # Setting auto_adjust=False might prevent the MultiIndex/renaming issue sometimes,
    # but let's handle the structure returned by yfinance robustly.
    df = yf.download(ticker, start=start, end=end, interval='1d')
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}. Check ticker symbol or date range.")

    # --- FIX 1: Check and Flatten MultiIndex Columns ---
    if isinstance(df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Flattening...")
        # Flatten MultiIndex by joining levels with an underscore
        # Example: ('Adj Close', '') -> 'adj_close_'
        # Ensure all parts of the tuple are strings before joining
        df.columns = ['_'.join(map(str, col)).strip().rstrip('_') for col in df.columns.values]
        print(f"Flattened columns: {df.columns.tolist()}")
    # --- End MultiIndex Handling ---

    # Clean column names (now guaranteed to be strings)
    # Replace spaces and hyphens with underscores, convert to lower case
    df.columns = [str(col).lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    print(f"Cleaned columns: {df.columns.tolist()}")

    # --- FIX 2: Rename specific columns back for 'ta' library compatibility ---
    # Create a mapping from potentially ticker-suffixed names to standard names
    ticker_lower = ticker.lower()
    rename_map = {
        f'open_{ticker_lower}': 'open',
        f'high_{ticker_lower}': 'high',
        f'low_{ticker_lower}': 'low',
        f'close_{ticker_lower}': 'close',
        f'volume_{ticker_lower}': 'volume',
        # Handle potential 'adj_close' if it exists and was flattened
        f'adj_close_{ticker_lower}': 'adj_close'
    }
    # Only keep mappings for columns that actually exist in the DataFrame
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}

    if actual_rename_map:
         print(f"Renaming columns for TA compatibility: {actual_rename_map}")
         df.rename(columns=actual_rename_map, inplace=True)
         print(f"Columns after renaming for TA: {df.columns.tolist()}")
    # --- End Renaming Fix ---

    # Ensure required columns exist for 'ta' library (This check should now pass)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        # Provide more context in error message
        raise ValueError(f"DataFrame must contain standard TA columns: {required_cols}. Missing: {missing_cols}. Available after potential renaming: {df.columns.tolist()}")

    # Calculate Technical Indicators using 'ta' library
    print("Calculating technical indicators using standard column names...")
    # Fill potential NaNs before calculating indicators if necessary
    # Using ffill().bfill() is generally safer than inplace=True
    df = df.ffill().bfill()

    # Add all TA features
    # Use error handling in case some indicators fail
    try:
        # Pass the DataFrame with standard column names ('open', 'high', etc.)
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
        )
    except Exception as e:
        print(f"Warning: Error calculating some TA features: {e}. Proceeding with available features.")


    # Drop rows with initial NaNs created by indicators with lookback periods
    # dropna() from ta.utils might be safer than pandas dropna for TA features
    # df = dropna(df) # Alternative using ta.utils.dropna
    df.dropna(axis=0, how='any', inplace=True) # Using pandas dropna

    print(f"Dataframe columns after adding indicators: {df.columns.tolist()}")

    # Select only numeric columns for scaling (important!)
    # Exclude boolean or object columns that might be added by 'ta'
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(f"Using {len(numeric_cols)} numeric features.")
    if not numeric_cols:
        raise ValueError("No numeric columns found after calculating technical indicators.")

    # Ensure 'close' column exists for target creation before creating df_numeric
    if 'close' not in df.columns:
         raise ValueError("'close' column not found after TA calculation and renaming. Check column list.")

    df_numeric = df[numeric_cols].copy() # Create a copy to avoid SettingWithCopyWarning

    # Handle potential infinity values if any indicator calculation resulted in them
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill remaining NaNs robustly
    df_numeric = df_numeric.ffill().bfill()
    # Check if any NaNs still exist after filling
    if df_numeric.isnull().values.any():
         print("Warning: NaNs still exist in numeric features after ffill/bfill. Consider further cleaning.")
         # Option: Drop rows/cols with NaNs, or use imputation
         df_numeric.dropna(inplace=True) # Drop rows with remaining NaNs as a fallback


    # Add target variable (shifted close price) - Use the standard 'close' price column
    df['target'] = df['close'].shift(-config.PREDICT_AHEAD)

    # Align numeric features and target by index, drop rows with NaN target
    common_index = df.index.intersection(df_numeric.index)
    df = df.loc[common_index]
    df_numeric = df_numeric.loc[common_index]

    # Drop rows where the target is NaN (the last few rows)
    df = df.dropna(subset=['target'])
    # Ensure alignment after dropping NaNs from target
    df_numeric = df_numeric.loc[df.index]

    if df_numeric.empty or df.empty:
        raise ValueError("DataFrame became empty after processing NaNs or aligning target.")


    print(f"Final feature shape: {df_numeric.shape}, Target shape: ({len(df['target'])},)")
    # Return df_numeric (features) and the target column from the original df
    return df_numeric, df['target']

# Example usage (optional, for testing)
if __name__ == '__main__':
    import config # Import config when running directly
    try:
        features, target = fetch_stock_data(config.STOCK_TICKER, config.START_DATE, config.END_DATE)
        print("\n--- Sample Features ---")
        print(features.head())
        print("\n--- Sample Target ---")
        print(target.head())
        print(f"\nFeature shape: {features.shape}")
        print(f"Target shape: {target.shape}")
    except ValueError as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

