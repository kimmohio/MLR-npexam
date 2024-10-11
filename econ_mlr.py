import pandas as pd
import statsmodels.api as sm

# Load the CSV file
file_path = 'econ.csv'
df = pd.read_csv(file_path)

# Define the dependent and independent variables
X = df[['CRUDE', 'INTEREST', 'FOREIGN', 'CONSUMER', 'GNP', 'PURCHASE']]
y = df['DJIA']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Get the R-squared value
r_squared = model.rsquared

# Get the regression equation coefficients
coefficients = model.params

# Display the R-squared value and regression equation
print(f'R-squared: {r_squared}')
print('Regression equation coefficients:')
print(coefficients)
