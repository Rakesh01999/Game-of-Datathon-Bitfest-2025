import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
import ast
from sklearn.exceptions import NotFittedError
import warnings
import logging
from typing import List, Dict, Any, Tuple

class JobMatchingPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('JobMatchingPipeline')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def safely_convert_list(self, string: str) -> int:
        """Safely convert string representation of list to length."""
        try:
            return len(ast.literal_eval(string)) if pd.notnull(string) else 0
        except (ValueError, SyntaxError):
            return 0
            
    def preprocess_data(self, dataset: pd.DataFrame, irrelevant_columns: List[str]) -> pd.DataFrame:
        """Preprocess the dataset by removing irrelevant columns and handling list-like columns."""
        self.logger.info("Starting data preprocessing...")
        
        # Drop irrelevant columns
        dataset = dataset.drop(columns=irrelevant_columns)
        
        # Handle list-like columns
        dataset['skills_length'] = dataset['skills'].apply(self.safely_convert_list)
        dataset = dataset.drop(columns=['skills'])
        
        # Handle missing values with more sophisticated strategies
        for col in dataset.columns:
            if dataset[col].isnull().any():
                null_pct = dataset[col].isnull().mean()
                if null_pct > 0.5:
                    self.logger.warning(f"Column {col} has {null_pct:.2%} missing values")
                
                if dataset[col].dtype == 'object':
                    # For categorical columns, add 'missing' category
                    dataset[col] = dataset[col].fillna('missing')
                else:
                    # For numerical columns, use median for skewed distributions
                    if dataset[col].skew() > 1:
                        dataset[col] = dataset[col].fillna(dataset[col].median())
                    else:
                        dataset[col] = dataset[col].fillna(dataset[col].mean())
        
        return dataset
        
    def create_stacking_model(self) -> StackingRegressor:
        """Create a stacking model with multiple base estimators."""
        estimators = [
            ('lgbm', LGBMRegressor(random_state=self.random_state)),
            ('rf', RandomForestRegressor(random_state=self.random_state)),
            ('ridge', Ridge(random_state=self.random_state))
        ]
        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(random_state=self.random_state),
            cv=5
        )
        
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create feature preprocessing pipeline."""
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=['number']).columns
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        
        return ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])
            
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, Any]]:
        """Train the model with cross-validation and hyperparameter tuning."""
        self.logger.info("Starting model training...")
        
        # Create preprocessing and model pipeline
        preprocessor = self.create_preprocessor(X)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', self.create_stacking_model())
        ])
        
        # Define parameter grid for hyperparameter tuning
        param_dist = {
            'regressor__lgbm__num_leaves': [31, 50, 70, 100],
            'regressor__lgbm__learning_rate': [0.01, 0.05, 0.1],
            'regressor__lgbm__n_estimators': [100, 200, 500],
            'regressor__lgbm__max_depth': [3, 5, 10, -1],
            'regressor__lgbm__min_child_samples': [10, 20, 30],
            'regressor__lgbm__subsample': [0.7, 0.8, 0.9],
            'regressor__rf__n_estimators': [100, 200],
            'regressor__rf__max_depth': [5, 10, None],
            'regressor__ridge__alpha': [0.1, 1.0, 10.0]
        }
        
        # Custom scoring metric
        custom_scorer = make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
        )
        
        # Perform randomized search with cross-validation
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=50,
            scoring=custom_scorer,
            cv=5,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        random_search.fit(X_train, y_train)
        
        # Evaluate model performance
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Best model MSE: {mse:.4f}")
        self.logger.info(f"Best model R2 score: {r2:.4f}")
        
        return best_model, random_search.best_params_
        
    def plot_feature_importance(self, model: Pipeline, X: pd.DataFrame) -> None:
        """Plot feature importance for the trained model."""
        try:
            import matplotlib.pyplot as plt
            
            # Get feature names after preprocessing
            feature_names = (model.named_steps['preprocessor']
                           .get_feature_names_out())
            
            # Get feature importances from the LGBM model
            lgbm_model = model.named_steps['regressor'].named_estimators_['lgbm']
            importances = lgbm_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances (LGBM)')
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), feature_names[indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.show()
            
        except NotFittedError:
            self.logger.error("Model not fitted. Please train the model first.")
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = JobMatchingPipeline(random_state=42)
    
    # Load and preprocess data
    dataset = pd.read_csv('train.csv')
    irrelevant_columns = [
        'address', 'career_objective', 'extra_curricular_activity_types',
        'extra_curricular_organization_names', 'extra_curricular_organization_links',
        'role_positions', 'languages', 'proficiency_levels',
        'certification_providers', 'certification_skills', 'online_links',
        'issue_dates', 'expiry_dates', 'age_requirement'
    ]
    
    processed_data = pipeline.preprocess_data(dataset, irrelevant_columns)
    
    # Prepare features and target
    X = processed_data.drop(columns=['matched_score'])
    y = processed_data['matched_score']
    
    # Train model
    best_model, best_params = pipeline.train_model(X, y)
    
    # Plot feature importance
    pipeline.plot_feature_importance(best_model, X)
    