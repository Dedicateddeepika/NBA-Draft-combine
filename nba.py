import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('nba_draft_combine_all_years.csv')

df_cleaned = df.dropna(subset=['Height (With Shoes)', 'Wingspan', 'Vertical (Max)', 'Agility', 'Weight', 'Draft pick'])

plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Height (With Shoes)'], kde=True)
plt.title('Distribution of Player Height (With Shoes)')
plt.xlabel('Height (With Shoes)')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Wingspan'], kde=True)
plt.title('Distribution of Player Wingspan')
plt.xlabel('Wingspan')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Height (With Shoes)', y='Draft pick', data=df_cleaned)
plt.title('Height vs Draft Pick')
plt.xlabel('Height (With Shoes)')
plt.ylabel('Draft Pick (lower = better)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Wingspan', y='Draft pick', data=df_cleaned)
plt.title('Wingspan vs Draft Pick')
plt.xlabel('Wingspan')
plt.ylabel('Draft Pick (lower = better)')
plt.show()

corr_matrix = df_cleaned[['Height (With Shoes)', 'Wingspan', 'Vertical (Max)', 'Agility', 'Weight', 'Draft pick']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of NBA Draft Attributes')
plt.show()

X = df_cleaned[['Height (With Shoes)', 'Wingspan', 'Vertical (Max)', 'Agility', 'Weight']]
y = df_cleaned['Draft pick']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.title('Predicted vs Actual Draft Pick')
plt.xlabel('Actual Draft Pick')
plt.ylabel('Predicted Draft Pick')
plt.show()

coefficients = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
