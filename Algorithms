
import os
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt




def convert_volume(vol_str):
    """Converts volume strings (e.g., '121.90K', '691.49M') to float."""
    if isinstance(vol_str, (int, float)): # Return if already numeric
        return float(vol_str)
    vol_str = str(vol_str).strip().upper() # Ensure string, remove whitespace, handle case
    
    if 'B' in vol_str:
        return float(vol_str.replace('B', '')) * 1_000_000_000
    elif 'M' in vol_str:
        return float(vol_str.replace('M', '')) * 1_000_000
    elif 'K' in vol_str:
        return float(vol_str.replace('K', '')) * 1_000
    else:
        try:
            # Handle cases with no suffix (just a number)
            return float(vol_str)
        except ValueError:
            # Handle unexpected formats or potential NaNs gracefully
            print(f"Warning: Could not convert volume value: {vol_str}. Returning NaN.")
            return np.nan


#Base Preprocessing Class
class Preprocessing:

    def __init__(self, dataset_path):
        #Load the dataset
        try:
            self.df = pd.read_csv(dataset_path)
            print(f"Dataset loaded successfully from {dataset_path}")
            # Convert 'Date' column to datetime objects right after loading
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            # Sort by date to ensure correct lagging and identification of the last row
            self.df = self.df.sort_values(by='Date').reset_index(drop=True)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}")
            raise
        except Exception as e:
            print(f"Error loading or parsing dataset: {e}")
            raise

        self.scaler = None # To store the feature scaler
        self.last_original_features_for_prediction = None # To store the row for future prediction
        self.feature_names = None # Store feature names after preprocessing
        
    def preprocess_data(self):

        # --- Data Cleaning and Type Conversion ---
        print("Starting data cleaning and type conversion...")
        # Handle 'Change %' - remove '%', convert to float, divide by 100
        if 'Change %' in self.df.columns:
            try:
                self.df['Change %'] = self.df['Change %'].astype(str).str.replace('%', '', regex=False)
                self.df['Change %'] = pd.to_numeric(self.df['Change %'], errors='coerce') / 100.0
                
            except Exception as e:
                print(f"Error processing 'Change %': {e}. Skipping column.")
                self.df.drop('Change %', axis=1, inplace=True, errors='ignore') # Drop if problematic
        else:
            print("Column 'Change %' not found.")


        # Handle 'Vol.' using the convert_volume function
        if 'Vol.' in self.df.columns:
            original_vol_type = self.df['Vol.'].dtype
            print(f"Processing 'Vol.' column (original type: {original_vol_type})...")
            self.df['Vol.'] = self.df['Vol.'].apply(convert_volume)
            # Check conversion results
            if self.df['Vol.'].isnull().any():
                 print(f"Warning: Some 'Vol.' values resulted in NaN after conversion.")
            print("'Vol.' processed.")
        else:
             print("Column 'Vol.' not found.")


        # Handle Price, Open, High, Low - remove commas, convert to float
        price_cols = ['Price', 'Open', 'High', 'Low']
        for col in price_cols:
            if col in self.df.columns:
                try:
                    # Ensure the column is treated as string before replacing commas
                    self.df[col] = self.df[col].astype(str).str.replace(',', '', regex=False)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except Exception as e:
                    print(f"Error processing column '{col}': {e}. Skipping column.")
                    self.df.drop(col, axis=1, inplace=True, errors='ignore') # Drop if problematic
            else:
                print(f"Column '{col}' not found.")

        # --- Feature Engineering (Lags) ---
        print("Creating lagged features...")
        # Define columns to lag (ensure they exist after cleaning)
        cols_to_lag = [col for col in ['Price', 'Change %', 'Vol.', 'High', 'Low', 'Open'] if col in self.df.columns]
       
        n_lags = 14
        for i in range(1, n_lags + 1):
            for col in cols_to_lag:
                 self.df[f'{col}_lag_{i}'] = self.df[col].shift(i)
        print(f"Created lags up to {n_lags} days for columns: {cols_to_lag}")

        # --- Store the last row for future prediction ---
        # This should be the row with the latest 'Date'
        # It contains the most recent actual values needed to predict the next day

        self.last_original_row_for_prediction = self.df.iloc[-1:].copy()
        print(f"Stored the last row (Date: {self.last_original_row_for_prediction['Date'].iloc[0]}) for future prediction input.")
       

        # --- Handle Missing Values ---
        initial_rows = len(self.df)
        # Drop rows with NaN values resulting from lagged features or initial NaNs/conversion errors
        self.df.dropna(inplace=True)
        rows_after_dropna = len(self.df)


        # --- Define Features (X) and Target (y) ---
        self.y = self.df['Price']

        # Features are all columns except 'Date' and 'Price'
        self.x = self.df.drop(['Date', 'Price'], axis=1)
        self.feature_names = self.x.columns.tolist() # Store feature names
        print(f"Preprocessing complete. X shape: {self.x.shape}, y shape: {self.y.shape}")

       

        return self.x, self.y


    def scale_features(self):
        self.scaler = StandardScaler()
        # Fit on the entire feature set X 
        self.x_scaled = self.scaler.fit_transform(self.x)
        print("Feature scaling complete using StandardScaler.")
        # Return the scaled features
        return self.x_scaled

    def get_last_features_for_prediction(self):
      
        # Select only the feature columns from the last stored row
        last_features_raw = self.last_original_row_for_prediction[self.feature_names].copy()

        # Scale these features using the *already fitted* scaler
        last_features_scaled = self.scaler.transform(last_features_raw) # Use transform, not fit_transform

        print(f"Prepared and scaled features for next-day prediction. Shape: {last_features_scaled.shape}")
        return last_features_scaled

    def split_data(self, test_size = 0.10, random_state=42, shuffle=True): # Changed shuffle to False for time series
        # Use the scaled features (self.x_scaled)
    

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_scaled, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
    
        print(f"Data split complete.\n")
        return self.x_train, self.x_test, self.y_train, self.y_test




#Base class for machine learning models
class Model_(Preprocessing):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.model = None # Placeholder for the actual model instance
        self.history = None # For storing training history (e.g., from Keras)
        self.next_day_prediction = None # Store the future prediction result

        # --- Core Preprocessing Steps ---
        self.x, self.y = self.preprocess_data()
        self.x_scaled = self.scale_features() # This now returns the scaled data
        # Prepare the feature vector for predicting the day *after* the dataset ends
        self.last_scaled_features = self.get_last_features_for_prediction()

        # --- Splitting Data ---
        # Use the scaled data for splitting. Crucially, turn OFF shuffling for time series.
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(shuffle=True)

        # --- Ensure y_train and y_test are float ---
        self.y_train = self.y_train.astype(float)
        self.y_test = self.y_test.astype(float)

    def evaluate(self):

        output_dir = self.get_output_dir()
        print(f"\n--- Evaluating {self.__class__.__name__} ---")
        print(f"Using test data: X_test shape {self.x_test.shape}, y_test shape {self.y_test.shape}")

        # --- Predict on Test Set ---
        
        y_pred = self.predict(self.x_test)   
        # Ensure y_pred is a flat array for metrics calculation
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
                y_pred = y_pred.flatten()

        # --- Calculate Regression Metrics ---
        try:
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            print(f"Evaluation Metrics:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2 Score: {r2:.4f}")

            metrics_contents = (f'Type: Regression\n'
                                f'Mean Squared Error (MSE): {mse:.4f}\n'
                                f'Root Mean Squared Error (RMSE): {rmse:.4f}\n'
                                f'Mean Absolute Error (MAE): {mae:.4f}\n'
                                f'R-squared (R2): {r2:.4f}\n')
            metrics_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

            # Save metrics to file
            metrics_path = os.path.join(output_dir, "metrics.txt")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as file:
                file.write(metrics_contents)
            print(f'Metrics saved to {metrics_path}')

        except Exception as e:
            print(f"Error calculating or saving metrics: {e}")
            traceback.print_exc()
            metrics_dict = None # Indicate failure
        
        
        # --- Plot Actual vs Predicted ---
        try:
            plt.figure(figsize=(7, 6)) # Slightly larger figure
            plt.scatter(self.y_test, y_pred, alpha=0.6, edgecolors='k', s=50) # Added edgecolors and size
            # Plot the ideal fit line based on actual data range
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], '--r', lw=2, label='Ideal Fit (y=x)')
            plt.xlabel('Actual Price ($)', fontsize=12)
            plt.ylabel('Predicted Price ($)', fontsize=12)
            plt.title(f'{self.__class__.__name__}: Actual vs. Predicted Prices', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout() # Adjust layout
            plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved Actual vs Predicted plot to: {plot_path}")
        except Exception as e:
            print(f"Error generating or saving Actual vs Predicted plot: {e}")
            traceback.print_exc()
   

        # --- Predict Future Day ---
        print("\n--- Predicting Next Day's Price ---")
        # Use the predict_future method which should be implemented by subclasses
        self.next_day_prediction = self.predict_future(n_days=1)
        if self.next_day_prediction is not None:
            # Ensure it's a single value if prediction was successful
            if isinstance(self.next_day_prediction, (np.ndarray, list)):
                self.next_day_prediction = self.next_day_prediction[0] # Take the first element

            print(f"Predicted price for the next day: ${self.next_day_prediction:,.2f}")

            # save this prediction to a file
            pred_path = os.path.join(output_dir, "next_day_prediction.txt")
            with open(pred_path, 'w') as f:
                f.write(f"Predicted Price for Next Day: {self.next_day_prediction:.4f}\n")
            print(f"Next day prediction saved to: {pred_path}")

        else:
            print("Failed to generate next day prediction.")

    

        return metrics_dict # Return metrics calculated on the test set


    def get_output_dir(self):
        """Gets the algorithm-specific output directory."""
        base_output_dir = "all_outputs"
        algo_name = self.__class__.__name__
        specific_dir = os.path.join(base_output_dir, f"{algo_name.replace(' ', '')}_output")
        os.makedirs(specific_dir, exist_ok=True) # Create dir if it doesn't exist
        return specific_dir




#Random Forrest class
class RandomForest(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        print("RandomForestRegressor initialized.")

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_future(self, n_days=1):
        if self.model is None:
             print("Error: RandomForest model is not trained.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: RandomForest future prediction currently only supports n_days=1.")
            # Add logic for multi-step prediction if needed later (more complex)

        # Predict using the last known scaled features
        prediction = self.model.predict(self.last_scaled_features)
        # Return the single predicted value (or first value if array)
        return prediction[0] if isinstance(prediction, np.ndarray) else prediction
       

#SVM class
class SupportVectorMachine(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn import svm

        C = 1.0
        epsilon = 0.1

        self.model = svm.SVR(kernel = 'linear', C=C, epsilon=epsilon, cache_size=500)

    def train(self):
        print("Training Support Vector Regressor...")
        self.model.fit(self.x_train, self.y_train)
        print("SVR Training Complete.")

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def predict_future(self, n_days=1):
        if self.model is None:
             print("Error: SVR model is not trained.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: SVR future prediction currently only supports n_days=1.")

        # Predict using the last known scaled features
        prediction = self.model.predict(self.last_scaled_features)
        return prediction[0] if isinstance(prediction, np.ndarray) else prediction
  


# Linear Regression class
class LinearRegression(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.linear_model import LinearRegression  # Import Linear Regression model
        
        self.model = LinearRegression()  # Initialize the model
    
    def train(self):
        self.model.fit(self.x_train, self.y_train)  # Train the linear regression model
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def predict_future(self, n_days=1):
        if self.model is None:
             print("Error: LinearRegression model is not trained.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: LinearRegression future prediction currently only supports n_days=1.")

        # Predict using the last known scaled features
        prediction = self.model.predict(self.last_scaled_features)
        return prediction[0] if isinstance(prediction, np.ndarray) else prediction
      
