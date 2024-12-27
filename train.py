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
