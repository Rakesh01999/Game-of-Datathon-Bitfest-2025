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

