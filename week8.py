import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load the data
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',10)

df = pd.read_csv("covid_19_india.csv")
df = df.drop(['Sno', 'Date', 'Time', 'State/UnionTerritory'], axis=1)

df.replace('-', pd.NA, inplace=True)
df.fillna(0, inplace=True)

# Features: Select relevant columns for predicting Confirmed cases (ConfirmedIndianNational, ConfirmedForeignNational, Cured, Deaths)
X = df[['ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', ]]
y = df['Confirmed']  # Target variable is the 'Confirmed' column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths']])
y = df['Confirmed']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Choose and train the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Cross-Validation (optional, to validate model performance)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation R-squared scores: {cv_scores}")
print(f"Mean Cross-Validation R-squared score: {cv_scores.mean()}")

# Visualize the actual vs predicted values (optional)
plt.plot(y_test.values.reset_index(drop=True), label="Actual Confirmed Cases", color='blue')
plt.plot(y_pred, label="Predicted Confirmed Cases", color='red')
plt.xlabel("Index")
plt.ylabel("Confirmed Cases")
plt.title("Actual vs Predicted Confirmed Cases")
plt.legend()
plt.show()
