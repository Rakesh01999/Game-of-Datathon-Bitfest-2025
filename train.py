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

