# NBA-Draft-combine
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset (replace 'nba_draft_data.csv' with your actual dataset path)
df = pd.read_csv('nba_draft_data.csv')

# Step 2: Data Cleaning and Preparation
# Inspect the data
print(df.info())
print(df.describe())

# Handle missing values (if any)
df = df.dropna()  # Simple drop, or use fillna() for more complex strategies

# Step 3: Exploratory Data Analysis (EDA)
# 3.1 Visualize distributions of key attributes
plt.figure(figsize=(10, 6))
sns.histplot(df['height (with shoes)'], kde=True)
plt.title('Distribution of Player Height (with shoes)')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['wingspan'], kde=True)
plt.title('Distribution of Player Wingspan')
plt.show()

# 3.2 Visualize relationships between variables (scatter plots)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='height (with shoes)', y='draftpick', data=df)
plt.title('Height vs Draft Pick')
plt.xlabel('Height (with shoes)')
plt.ylabel('Draft Pick (lower = better)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='wingspan', y='draftpick', data=df)
plt.title('Wingspan vs Draft Pick')
plt.xlabel('Wingspan')
plt.ylabel('Draft Pick (lower = better)')
plt.show()

# Step 4: Correlation Analysis
# Create a correlation matrix
corr_matrix = df.corr()

# Visualize correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of NBA Draft Attributes')
plt.show()

# Step 5: Build a Regression Model
# Predict draft pick based on physical and athletic metrics
# Select features for the model
X = df[['height (with shoes)', 'wingspan', 'vertical (max)', 'agility', 'weight']]
y = df['draftpick']

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Step 6: Evaluate the Model
# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Visualize Predictions vs Actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Predicted vs Actual Draft Pick')
plt.xlabel('Actual Draft Pick')
plt.ylabel('Predicted Draft Pick')
plt.show()

# Step 8: Feature Importance (Coefficient Analysis)
coefficients = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# This tells you how much each attribute contributes to draft pick prediction.
