import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

covid_df = pd.read_csv("covid_19_india.csv")

# Convert Date to datetime
covid_df['Date'] = pd.to_datetime(covid_df['Date'])

# Group by date and sum the cases for all states
daily_cases = covid_df.groupby('Date')['Confirmed'].sum().reset_index()

# Feature engineering
daily_cases['Days'] = (daily_cases['Date'] - daily_cases['Date'].min()).dt.days

# Prepare the data for modeling
X = daily_cases[['Days']]
y = daily_cases['Confirmed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Visualize the results
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Days since first case')
plt.ylabel('Cumulative Confirmed Cases')
plt.title('COVID-19 Cases in India: Actual vs Predicted')
plt.legend()
plt.show()

# Make future predictions
last_day = daily_cases['Days'].max()
future_days = pd.DataFrame({'Days': range(last_day + 1, last_day + 31)})
future_predictions = model.predict(future_days)

print("\nPredictions for the next 30 days:")
for day, prediction in zip(future_days['Days'], future_predictions):
    future_date = daily_cases['Date'].min() + pd.Timedelta(days=int(day))
    print(f"{future_date.date()}: {prediction:.0f} cases")