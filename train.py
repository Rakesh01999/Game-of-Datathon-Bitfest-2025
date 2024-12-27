from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import pandas as pd

# Load and Inspect the Dataset
dataset = pd.read_csv('train.csv')
print(dataset.head())  # Inspect the first few rows
print(dataset.info())  # Check data types and missing values

# Drop Irrelevant Columns
irrelevant_columns = [
    'address', 'career_objective', 'extra_curricular_activity_types',
    'extra_curricular_organization_names', 'extra_curricular_organization_links',
    'role_positions', 'languages', 'proficiency_levels',
    'certification_providers', 'certification_skills', 'online_links',
    'issue_dates', 'expiry_dates', 'age_requirement'
]
dataset.drop(columns=irrelevant_columns, inplace=True)
print(dataset.head())  # Confirm columns are dropped

# Handle Missing Values
for col in dataset.columns:
    if dataset[col].isnull().any():
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)
print(dataset.isnull().sum())  # Check remaining missing values

# Define Features and Target
target_column = 'matched_score'
X = dataset.drop(columns=[target_column])
y = dataset[target_column]

print("Features:", X.head())
print("Target:", y.head())

# Preprocess Data
object_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns
print("Object Columns:", object_columns)
print("Numerical Columns:", numerical_columns)

# Define preprocessors:
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, object_columns)
    ])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Size:", X_train.shape)
print("Test Size:", X_test.shape)

# Create Base Model
base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMRegressor(random_state=42))
])
base_model.fit(X_train, y_train)

y_pred = base_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Base Model):", mse)


# Corrected GridSearchCV Block
# 1. Data Quality Issues
# Check for Null or NaN Values: 
print(dataset.isnull().sum())
# Steps to Resolve the Issue:
# Check Data Types of Columns:
numeric_columns = dataset.select_dtypes(include=['number']).columns
print(numeric_columns)
# Filter Numeric Columns: Use the numeric columns for .var():
numeric_data = dataset[numeric_columns]
print(numeric_data.var())
# Handle String Columns: 
dataset['skills_count'] = dataset['skills'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
# Fill or Remove Missing Data
dataset.fillna({'skills': '[]'}, inplace=True)
# Debugging Warnings:
dataset.loc[:, 'skills'] = dataset['skills'].fillna('[]')
# Train-Test Split:
# ...

print(dataset.columns)
print("Columns after dropping irrelevant ones:", dataset.columns)
if 'matched_score' in dataset.columns:
    print(dataset['matched_score'].head())
else:
    print("Target column 'matched_score' is missing!")
print(dataset['matched_score'].isnull().sum())
target_column = 'matched_score'  # Use exact column name from dataset


# Ensure Target Column Exists
assert 'matched_score' in dataset.columns, "Target column 'matched_score' is missing!"

# Check Numeric Columns
numeric_columns = dataset.select_dtypes(include=['number']).columns
print("Numeric Columns:", numeric_columns)

# Split Data
X = dataset[numeric_columns].drop(columns=['matched_score'], errors='ignore')  # Avoid KeyError
y = dataset['matched_score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shapes -> X_train:", X_train.shape, "y_train:", y_train.shape)
