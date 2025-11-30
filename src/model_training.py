"""
Model Training Module
Trains and evaluates fraud detection models
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Tuple, Dict

class FraudModelTrainer:
    """Train and evaluate fraud detection models"""

    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Initialize model trainer

        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of test data
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def split_data(self) -> None:
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=42, 
            stratify=self.y
        )

        print(f"✓ Train set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        print(f"✓ Fraud in train: {self.y_train.sum()} ({self.y_train.sum()/len(self.y_train)*100:.1f}%)")
        print(f"✓ Fraud in test: {self.y_test.sum()} ({self.y_test.sum()/len(self.y_test)*100:.1f}%)")

    def scale_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler

        Returns:
            Scaled train and test sets
        """
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def train_random_forest(self, **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest model

        Returns:
            Trained Random Forest model
        """
        print("\nTraining Random Forest...")

        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        params.update(kwargs)

        rf_model = RandomForestClassifier(**params)
        rf_model.fit(self.X_train, self.y_train)

        self.models['Random Forest'] = rf_model
        self._evaluate_model('Random Forest', rf_model, self.X_test, self.y_test)

        return rf_model

    def train_gradient_boosting(self, **kwargs) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting model

        Returns:
            Trained Gradient Boosting model
        """
        print("\nTraining Gradient Boosting...")

        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        params.update(kwargs)

        gb_model = GradientBoostingClassifier(**params)
        gb_model.fit(self.X_train, self.y_train)

        self.models['Gradient Boosting'] = gb_model
        self._evaluate_model('Gradient Boosting', gb_model, self.X_test, self.y_test)

        return gb_model

    def train_logistic_regression(self, **kwargs) -> LogisticRegression:
        """
        Train Logistic Regression model

        Returns:
            Trained Logistic Regression model
        """
        print("\nTraining Logistic Regression...")

        # Scale features for logistic regression
        X_train_scaled, X_test_scaled = self.scale_features()

        params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        params.update(kwargs)

        lr_model = LogisticRegression(**params)
        lr_model.fit(X_train_scaled, self.y_train)

        self.models['Logistic Regression'] = lr_model
        self._evaluate_model('Logistic Regression', lr_model, X_test_scaled, self.y_test)

        return lr_model

    def _evaluate_model(self, name: str, model, X_test, y_test) -> None:
        """
        Evaluate model performance

        Args:
            name: Model name
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        self.results[name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }

        # Print results
        print(f"\n{name} Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

    def train_all_models(self) -> Dict:
        """
        Train all models

        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("🤖 TRAINING ALL MODELS")
        print("="*60)

        self.split_data()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_logistic_regression()

        return self.models

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for name, results in self.results.items():
            row = {'Model': name}
            row.update(results['metrics'])
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        print("\n" + "="*60)
        print("🏆 MODEL COMPARISON")
        print("="*60)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to file

        Args:
            model_name: Name of model to save
            filepath: Path to save model
        """
        if model_name in self.models:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            print(f"✓ Saved {model_name} to {filepath}")
        else:
            print(f"✗ Model {model_name} not found")

    def save_scaler(self, filepath: str) -> None:
        """
        Save scaler to file

        Args:
            filepath: Path to save scaler
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler to {filepath}")

    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based model

        Args:
            model_name: Name of model

        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print(f"\n🔍 Top 10 Features ({model_name}):")
            print(importance_df.head(10).to_string(index=False))

            return importance_df
        else:
            print(f"{model_name} does not have feature importance")
            return None
