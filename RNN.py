import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

df = pd.read_csv("Bitcoin_Data.csv")

#print("Date range of full dataset:", df.index.min(), "to", df.index.max())

#Convert 'Date' column to a datetime format for easier handling and sets it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#Sorts data in ascending order bassed on the data
df.sort_index(ascending=True, inplace=True)


# Convert 'Price' column to numeric and remove commas
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '', regex=True), errors='coerce')

#This function converts volume data (e.g., '121.90K', '691.49M') to numeric values.
#It handles different suffixes like K (thousands), M (millions), and B (billions).
def convert_volume(vol_str):
    if isinstance(vol_str, (int, float)):
        return float(vol_str)
    vol_str = str(vol_str).strip().upper()
    if 'B' in vol_str:
        return float(vol_str.replace('B', '')) * 1_000_000_000
    elif 'M' in vol_str:
        return float(vol_str.replace('M', '')) * 1_000_000
    elif 'K' in vol_str:
        return float(vol_str.replace('K', '')) * 1_000
    else:
        try:
            return float(vol_str)
        except ValueError:
            return np.nan

#Applies the convert_volume function to the 'Vol.' column to clean the volume data
df['Vol.'] = df['Vol.'].apply(convert_volume)

# Preprocessing: Scaling the data, by normalizing the 'Price' column between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
df['Price'] = scaler.fit_transform(df[['Price']])

#Lag is commonly used with dealing with time series data.
#Creating lagged feature (past prices) to use as input for LSTM model.
#the loop creates columns 'Price_lag_i' each containing the previous days price.
#shifts the 'Price' column by i steps, creating a lag of i days.
n_lags = 5
for i in range(1, n_lags + 1):
    df[f'Price_lag_{i}'] = df['Price'].shift(i)

# Drop NaN values introduced by lagging
df.dropna(inplace=True)

#print("Total rows after cleaning and lagging:", len(df))

#Splitting the data, 80% used for training, and 20% used for testing.
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

#This function converts the lagged data into features X and labels y for training the model.
# It iterates over the dataset and collects the lagged prices as features and the current price as the target variable.
#LSTM model with lagging
def create_dataset(data, lags):
    X, y = [], []
    lag_cols = [f'Price_lag_{i}' for i in range(1, lags + 1)]
    for i in range(len(data)):
        X.append(data[lag_cols].iloc[i].values)
        y.append(data['Price'].iloc[i])
    return np.array(X), np.array(y)

#Create training and testing datasets by calling create_dataset function.
X_train, y_train = create_dataset(train_data, lags=n_lags)
X_test, y_test = create_dataset(test_data, lags=n_lags)

#Reshaping the data for the LSTM model, which expects 3D input with the shape.
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Ensure the data is reshaped correctly
#print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#Building the LSTM model with 2 layers
model = Sequential([
    #The first LSTM layer has 50 units and returns sequences to pass the output to the next LSTM layer.
    LSTM(50, return_sequences=True, input_shape=(n_lags, 1)),
    #The second LSTM layer also has 50 units but does not return sequences because we only need a final prediction.
    LSTM(50, return_sequences=False),
    #A Dense layer with 25 neurons uses the ReLU activation function to introduce non-linearity.
    Dense(25, activation='relu'),
    #The final Dense layer outputs a single value, which represents the predicted Bitcoin price.
    Dense(1)
])

# Compiling the model with the adam optimizer and MSE as the loss function.
model.compile(optimizer='adam', loss='mean_squared_error')


#print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
#print(f"Data type of X_train: {type(X_train)}, y_train: {type(y_train)}")

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    #uses the test data as validation during training to monitor performance on unseen data.
    validation_data=(X_test, y_test))

#Model predicts the price on test set
predictions = model.predict(X_test)

# Reverse scaling for predictions to return predections to its original range.
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluates the Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, and the R^2.
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

print("\nEvaluation Metrics on Test Data:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print("\n")


# Get the correct index after dropping rows due to lagging
plot_index = df.iloc[-len(y_test):].index
#print(f"Length of plot_index: {len(plot_index)}")
#print(f"Length of actual_prices: {len(actual_prices)}")
#print(f"Length of predicted_prices: {len(predicted_prices)}")

#Plotting actual v predicted over time for comparison.
plt.figure(figsize=(10, 6))
plt.plot(plot_index, actual_prices, label='Actual Price', color='blue')
plt.plot(plot_index, predicted_prices, label='Predicted Price', color='red')

# Plot the last actual price point
last_date = df.index[-1]
last_price_scaled = df['Price'].iloc[-1]
last_price_actual = scaler.inverse_transform([[last_price_scaled]])[0][0]
plt.scatter(last_date, last_price_actual, color='black', label='Last Actual Price', zorder=5)
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#print the last actual price
print(f"Last actual price on {last_date.date()}: ${last_price_actual:,.2f}")

#Predict the next day price
# Get last n_lags prices from df['Price']
last_prices = df['Price'].values[-n_lags:]
X_next = last_prices.reshape((1, n_lags, 1))

#Predict next day
next_day_scaled = model.predict(X_next)
next_day_price = scaler.inverse_transform(next_day_scaled)

#Date of prediction
next_date = df.index[-1] + timedelta(days=1)
print(f"\n📈 Predicted Bitcoin price for {next_date.date()}: ${next_day_price[0][0]:,.2f}")
