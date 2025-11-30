"""
Fraud Predictor Module
Production-ready fraud detection system
Author: Esprit AI Student
"""

import pandas as pd
import pickle
from typing import Dict, Union

class FraudPredictor:
    """Real-time fraud prediction system"""

    def __init__(self):
        """Initialize fraud predictor"""
        self.model = None
        self.scaler = None
        self.feature_names = [
            'DAYS_TO_DECLARE', 'SAME_DAY_DECLARATION', 'LATE_DECLARATION',
            'VERY_LATE_DECLARATION', 'ACCIDENT_MONTH', 'ACCIDENT_YEAR',
            'ACCIDENT_DAY_OF_WEEK', 'IS_WEEKEND', 'RESPONSABILITE',
            'ZERO_RESPONSIBILITY', 'FULL_RESPONSIBILITY', 'VAGUE_LOCATION',
            'HIGH_RISK_LOCATION', 'IS_MATERIAL_DAMAGE', 'MISSING_GARAGE',
            'HAS_CONSTAT', 'EXPERT_FREQUENCY', 'HIGH_FREQUENCY_EXPERT',
            'GARAGE_FREQUENCY'
        ]

    def load_model(self, model_path: str) -> None:
        """
        Load trained model from file

        Args:
            model_path: Path to model file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded model from {model_path}")

    def load_scaler(self, scaler_path: str) -> None:
        """
        Load scaler from file

        Args:
            scaler_path: Path to scaler file
        """
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Loaded scaler from {scaler_path}")

    def predict(self, claim_data: Dict) -> Dict:
        """
        Predict if a claim is fraudulent

        Args:
            claim_data: Dictionary with claim features

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Create DataFrame
        df = pd.DataFrame([claim_data])

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns
        df = df[self.feature_names]

        # Scale if scaler is available
        if self.scaler:
            df_scaled = self.scaler.transform(df)
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0][1]
        else:
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
            risk_color = "🟢"
        elif probability < 0.7:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        else:
            risk_level = "HIGH"
            risk_color = "🔴"

        return {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'risk_icon': risk_color,
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium'
        }

    def predict_batch(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for multiple claims

        Args:
            claims_df: DataFrame with claims

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in claims_df.columns:
                claims_df[feature] = 0

        # Reorder columns
        df = claims_df[self.feature_names]

        # Predict
        if self.scaler:
            df_scaled = self.scaler.transform(df)
            predictions = self.model.predict(df_scaled)
            probabilities = self.model.predict_proba(df_scaled)[:, 1]
        else:
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)[:, 1]

        # Create results DataFrame
        results = claims_df.copy()
        results['FRAUD_PREDICTION'] = ['FRAUD' if p == 1 else 'LEGITIMATE' for p in predictions]
        results['FRAUD_PROBABILITY'] = probabilities * 100
        results['RISK_LEVEL'] = ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.3 else 'LOW' 
                                  for p in probabilities]

        return results

    def explain_prediction(self, claim_data: Dict, top_n: int = 5) -> Dict:
        """
        Explain why a claim was flagged as fraud

        Args:
            claim_data: Dictionary with claim features
            top_n: Number of top features to show

        Returns:
            Dictionary with explanation
        """
        prediction_result = self.predict(claim_data)

        # Get feature values
        risk_factors = []

        if claim_data.get('LATE_DECLARATION', 0) == 1:
            risk_factors.append("⚠️  Late claim declaration (>30 days)")

        if claim_data.get('VERY_LATE_DECLARATION', 0) == 1:
            risk_factors.append("🚨 Very late declaration (>60 days)")

        if claim_data.get('SAME_DAY_DECLARATION', 0) == 1:
            risk_factors.append("⚠️  Claim declared same day as accident")

        if claim_data.get('ZERO_RESPONSIBILITY', 0) == 1:
            risk_factors.append("⚠️  Claiming zero responsibility")

        if claim_data.get('VAGUE_LOCATION', 0) == 1:
            risk_factors.append("🚨 Vague or suspicious location")

        if claim_data.get('MISSING_GARAGE', 0) == 1:
            risk_factors.append("⚠️  Missing garage information")

        if claim_data.get('HIGH_FREQUENCY_EXPERT', 0) == 1:
            risk_factors.append("⚠️  High-frequency expert (potential collusion)")

        if claim_data.get('IS_WEEKEND', 0) == 1:
            risk_factors.append("ℹ️  Accident occurred on weekend")

        return {
            'prediction': prediction_result,
            'risk_factors': risk_factors[:top_n],
            'total_risk_factors': len(risk_factors)
        }


if __name__ == "__main__":
    # Example usage
    predictor = FraudPredictor()
    predictor.load_model('random_forest_model.pkl')

    # Suspicious claim
    suspicious_claim = {
        'DAYS_TO_DECLARE': 45,
        'LATE_DECLARATION': 1,
        'ZERO_RESPONSIBILITY': 1,
        'VAGUE_LOCATION': 1,
        'MISSING_GARAGE': 1,
        'HIGH_FREQUENCY_EXPERT': 1,
        'IS_WEEKEND': 1
    }

    result = predictor.explain_prediction(suspicious_claim)
    print("\nPrediction:", result['prediction'])
    print("Risk Factors:")
    for factor in result['risk_factors']:
        print(f"  {factor}")
