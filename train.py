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

# Dataset Inspection and Cleaning
import ast

# Ensure 'skills' column is a list-like structure
dataset['skills'] = dataset['skills'].fillna('[]').apply(ast.literal_eval)
dataset['skills_length'] = dataset['skills'].apply(len)

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

# Validate Irrelevant Columns Before Dropping
# Ensure irrelevant columns exist before dropping
missing_cols = [col for col in irrelevant_columns if col not in dataset.columns]
if missing_cols:
    print(f"Warning: The following columns are missing and cannot be dropped: {missing_cols}")
dataset.drop(columns=irrelevant_columns, errors='ignore', inplace=True)
# dataset.drop(columns=irrelevant_columns, inplace=True)


# Add Derived Feature
dataset['skills_length'] = dataset['skills'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

# Check the Dataset After Adding the Feature
print(dataset.head())

# Handle Missing Values
# for col in dataset.columns:
#     if dataset[col].isnull().any():
#         dataset[col].fillna(dataset[col].mode()[0], inplace=True)
# Fill Missing Values
# for col in dataset.columns:
#     if dataset[col].isnull().any():
#         if dataset[col].dtype == 'object':
#             dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
#         else:
#             dataset[col] = dataset[col].fillna(dataset[col].mean())

# Consolidate Missing Value Handling
# Ensure target column has no missing values
assert target_column in dataset.columns, "Target column is missing!"
if dataset[target_column].isnull().any():
    print(f"Filling missing values in target column '{target_column}' with its mean.")
    dataset[target_column].fillna(dataset[target_column].mean(), inplace=True)

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
# Plot histograms of numeric features to check for scaling needs
X_train[numerical_columns].hist(figsize=(10, 8))

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
# Base Model with Preprocessing Pipeline
# base_model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', LGBMRegressor(random_state=42))
# ])
# base_model.fit(X_train, y_train)
# y_pred = base_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error (Base Model):", mse)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base Model with Preprocessing Pipeline
base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMRegressor(random_state=42))
])

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [100, 200, 500],
    'classifier__num_leaves': [31, 50, 100],
    'classifier__max_depth': [-1, 10, 20]
}
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Evaluate the Best Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
optimized_mse = mean_squared_error(y_test, y_pred)
print("Optimized Mean Squared Error:", optimized_mse)


# Replace GridSearchCV with RandomizedSearchCV for efficiency
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=20,  # Limit number of parameter combinations
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# Evaluate Best Model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
optimized_mse = mean_squared_error(y_test, y_pred)
print("Optimized Mean Squared Error:", optimized_mse)


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


