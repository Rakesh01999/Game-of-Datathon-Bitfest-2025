from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
import pandas as pd

# Load the training dataset
dataset = pd.read_csv('train.csv')

# Drop irrelevant columns
irrelevant_columns = [
    'address', 'career_objective', 'extra_curricular_activity_types',
    'extra_curricular_organization_names', 'extra_curricular_organization_links',
    'role_positions', 'languages', 'proficiency_levels',
    'certification_providers', 'certification_skills', 'online_links',
    'issue_dates', 'expiry_dates', 'age_requirement'
]
dataset.drop(columns=irrelevant_columns, inplace=True)

# Fill missing values using mode
for col in dataset.columns:
    if dataset[col].isnull().any():
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# Identify the target column and separate features (X) and target (y)
target_column = 'matched_score'
X = dataset.drop(columns=[target_column])
y = dataset[target_column]

# Identify object and numerical columns
object_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

# Define preprocessors for categorical and numerical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())  # Handles outliers
])

# Combine preprocessors into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, object_columns)
    ])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine preprocessing and model training in a pipeline
base_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMRegressor(random_state=42))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [10, 20, -1],  # Use -1 for unlimited depth
    'classifier__num_leaves': [31, 50, 70],
    'classifier__subsample': [0.8, 1.0],  # Prevent overfitting
    'classifier__colsample_bytree': [0.8, 1.0],  # Use a subset of features
    'classifier__reg_alpha': [0, 0.1, 0.5],  # L1 regularization
    'classifier__reg_lambda': [0, 0.1, 0.5]  # L2 regularization
}

grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Retrieve the best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the optimized model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Optimized Mean Squared Error:", mse)

# Feature Importance (Optional)
classifier = best_model.named_steps['classifier']
if hasattr(classifier, 'feature_importances_'):
    feature_importances = classifier.feature_importances_
    print("Feature Importances:", feature_importances)

# Load the test dataset
test_dataset = pd.read_csv('test.csv')

# Drop the same columns as in the training dataset
test_dataset.drop(columns=irrelevant_columns, inplace=True)

# Fill NA values in test dataset using mode from the training dataset
for col in test_dataset.columns:
    if test_dataset[col].isnull().any():
        test_dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# Remove ID from test_dataset
test_X = test_dataset.drop(columns=['ID'])

# Make predictions on the test dataset
y_pred_test = best_model.predict(test_X)

# Print the predictions
print("Predictions on Test Data:", y_pred_test)
