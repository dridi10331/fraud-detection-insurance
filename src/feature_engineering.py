"""
Feature Engineering Module
Creates fraud detection features from insurance claims data
"""

import pandas as pd
import numpy as np

class FraudFeatureEngineer:
    """Engineer features for fraud detection"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer

        Args:
            df: Input DataFrame with insurance claims
        """
        self.df = df.copy()
        self.feature_names = []

    def create_time_features(self) -> None:
        """Create time-based fraud indicators"""
        print("Creating time-based features...")

        # Days to declare claim
        self.df['DAYS_TO_DECLARE'] = (
            self.df['DATE DECLARATION'] - self.df['DATE SINISTRE']
        ).dt.days

        # Same day declaration (suspicious for staged accidents)
        self.df['SAME_DAY_DECLARATION'] = (
            self.df['DAYS_TO_DECLARE'] == 0
        ).astype(int)

        # Late declaration (>30 days - potentially fabricated)
        self.df['LATE_DECLARATION'] = (
            self.df['DAYS_TO_DECLARE'] > 30
        ).astype(int)

        # Very late declaration (>60 days)
        self.df['VERY_LATE_DECLARATION'] = (
            self.df['DAYS_TO_DECLARE'] > 60
        ).astype(int)

        # Extract temporal features
        self.df['ACCIDENT_MONTH'] = self.df['DATE SINISTRE'].dt.month
        self.df['ACCIDENT_YEAR'] = self.df['DATE SINISTRE'].dt.year
        self.df['ACCIDENT_DAY_OF_WEEK'] = self.df['DATE SINISTRE'].dt.dayofweek
        self.df['IS_WEEKEND'] = self.df['ACCIDENT_DAY_OF_WEEK'].isin([5, 6]).astype(int)

        self.feature_names.extend([
            'DAYS_TO_DECLARE', 'SAME_DAY_DECLARATION', 'LATE_DECLARATION',
            'VERY_LATE_DECLARATION', 'ACCIDENT_MONTH', 'ACCIDENT_YEAR',
            'ACCIDENT_DAY_OF_WEEK', 'IS_WEEKEND'
        ])

        print(f"✓ Created {len(self.feature_names)} time features")

    def create_responsibility_features(self) -> None:
        """Create responsibility pattern features"""
        print("Creating responsibility features...")

        # Zero responsibility (always claiming not at fault)
        self.df['ZERO_RESPONSIBILITY'] = (
            self.df['RESPONSABILITE'] == 0
        ).astype(int)

        # Full responsibility
        self.df['FULL_RESPONSIBILITY'] = (
            self.df['RESPONSABILITE'] == 100
        ).astype(int)

        self.feature_names.extend([
            'RESPONSABILITE', 'ZERO_RESPONSIBILITY', 'FULL_RESPONSIBILITY'
        ])

        print(f"✓ Created responsibility features")

    def create_location_features(self) -> None:
        """Create location-based features"""
        print("Creating location features...")

        # Vague or fake locations
        self.df['VAGUE_LOCATION'] = self.df['LIEU SINISTRE'].isin([
            'XX XX', 'X', '*', '.', 'TN', 'UNKNOWN'
        ]).astype(int)

        # High risk locations (fraud hotspots)
        self.df['HIGH_RISK_LOCATION'] = self.df['LIEU SINISTRE'].isin([
            'SOUSSE', 'TUNIS', 'ARIANA'
        ]).astype(int)

        self.feature_names.extend(['VAGUE_LOCATION', 'HIGH_RISK_LOCATION'])

        print(f"✓ Created location features")

    def create_claim_type_features(self) -> None:
        """Create claim type features"""
        print("Creating claim type features...")

        # Material damage (easier to fake)
        self.df['IS_MATERIAL_DAMAGE'] = (
            self.df['TYPE SINISTRE'] == 'A:MATERIEL'
        ).astype(int)

        self.feature_names.append('IS_MATERIAL_DAMAGE')

        print(f"✓ Created claim type features")

    def create_documentation_features(self) -> None:
        """Create features based on missing/incomplete documentation"""
        print("Creating documentation features...")

        # Missing garage info
        self.df['MISSING_GARAGE'] = (
            self.df['CODE GARAGE'] == 0.0
        ).astype(int)

        # Has constat (police report)
        self.df['HAS_CONSTAT'] = self.df['CONSTAT'].notna().astype(int)

        self.feature_names.extend(['MISSING_GARAGE', 'HAS_CONSTAT'])

        print(f"✓ Created documentation features")

    def create_network_features(self) -> None:
        """Create features based on expert/garage networks"""
        print("Creating network features...")

        # Expert frequency (potential collusion)
        expert_counts = self.df['CODE EXPERT'].value_counts()
        self.df['EXPERT_FREQUENCY'] = self.df['CODE EXPERT'].map(expert_counts)
        self.df['HIGH_FREQUENCY_EXPERT'] = (
            self.df['EXPERT_FREQUENCY'] > 50
        ).astype(int)

        # Garage frequency
        garage_counts = self.df['CODE GARAGE'].value_counts()
        self.df['GARAGE_FREQUENCY'] = self.df['CODE GARAGE'].map(garage_counts)

        self.feature_names.extend([
            'EXPERT_FREQUENCY', 'HIGH_FREQUENCY_EXPERT', 'GARAGE_FREQUENCY'
        ])

        print(f"✓ Created network features")

    def create_fraud_label(self) -> None:
        """Create fraud label based on suspicious patterns"""
        print("Creating fraud labels...")

        # Calculate fraud score
        fraud_score = (
            self.df['SAME_DAY_DECLARATION'] * 2 +
            self.df['LATE_DECLARATION'] * 3 +
            self.df['VERY_LATE_DECLARATION'] * 4 +
            self.df['ZERO_RESPONSIBILITY'] * 2 +
            self.df['VAGUE_LOCATION'] * 3 +
            self.df['MISSING_GARAGE'] * 2 +
            self.df['HIGH_FREQUENCY_EXPERT'] * 2 +
            self.df['IS_MATERIAL_DAMAGE'] * 1
        )

        # Label as fraud if score > threshold (top 15% most suspicious)
        threshold = fraud_score.quantile(0.85)
        self.df['FRAUD_LABEL'] = (fraud_score > threshold).astype(int)

        fraud_count = self.df['FRAUD_LABEL'].sum()
        print(f"✓ Created fraud labels: {fraud_count} fraudulent ({fraud_count/len(self.df)*100:.1f}%)")

    def engineer_all_features(self) -> pd.DataFrame:
        """
        Run all feature engineering steps

        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*60)
        print("🔨 FEATURE ENGINEERING PIPELINE")
        print("="*60 + "\n")

        self.create_time_features()
        self.create_responsibility_features()
        self.create_location_features()
        self.create_claim_type_features()
        self.create_documentation_features()
        self.create_network_features()
        self.create_fraud_label()

        print(f"\n✅ Total features created: {len(self.feature_names)}")
        print(f"✅ Dataset shape: {self.df.shape}")

        return self.df

    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return self.feature_names


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to engineer all fraud detection features

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    engineer = FraudFeatureEngineer(df)
    return engineer.engineer_all_features()
