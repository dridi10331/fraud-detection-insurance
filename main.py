"""
Main Script - Complete Fraud Detection Pipeline
Esprit AI Project - 3rd Year Student
Author: Esprit AI Student
"""

import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_preprocess
from src.feature_engineering import engineer_fraud_features
from src.model_training import FraudModelTrainer
from src.fraud_predictor import FraudPredictor

def main():
    """Run complete fraud detection pipeline"""
    
    print("="*80)
    print("🚨 INSURANCE FRAUD DETECTION SYSTEM")
    print("🏫 Esprit - 3rd Year AI Student")
    print("="*80)
    
    # Step 1: Load and preprocess data
    print("\n📊 STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    df = load_and_preprocess('data/raw/FDS-3.csv')
    print(f"✓ Loaded and preprocessed {len(df)} claims")
    
    # Step 2: Feature engineering
    print("\n🔨 STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    df_engineered = engineer_fraud_features(df)
    print(f"✓ Engineered features, final shape: {df_engineered.shape}")
    
    # Save preprocessed data
    df_engineered.to_csv('preprocessed_claims.csv', index=False)
    print("✓ Saved preprocessed data to 'preprocessed_claims.csv'")
    
    # Step 3: Prepare data for modeling
    print("\n🎯 STEP 3: PREPARE MODELING DATA")
    print("="*80)
    
    feature_cols = [
        'DAYS_TO_DECLARE', 'SAME_DAY_DECLARATION', 'LATE_DECLARATION',
        'VERY_LATE_DECLARATION', 'ACCIDENT_MONTH', 'ACCIDENT_YEAR',
        'ACCIDENT_DAY_OF_WEEK', 'IS_WEEKEND', 'RESPONSABILITE',
        'ZERO_RESPONSIBILITY', 'FULL_RESPONSIBILITY', 'VAGUE_LOCATION',
        'HIGH_RISK_LOCATION', 'IS_MATERIAL_DAMAGE', 'MISSING_GARAGE',
        'HAS_CONSTAT', 'EXPERT_FREQUENCY', 'HIGH_FREQUENCY_EXPERT',
        'GARAGE_FREQUENCY'
    ]
    
    X = df_engineered[feature_cols].fillna(0)
    y = df_engineered['FRAUD_LABEL']
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Samples: {X.shape[0]}")
    print(f"✓ Fraud cases: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Step 4: Train models
    print("\n🤖 STEP 4: MODEL TRAINING")
    print("="*80)
    
    trainer = FraudModelTrainer(X, y, test_size=0.2)
    models = trainer.train_all_models()
    
    # Step 5: Compare models
    print("\n🏆 STEP 5: MODEL COMPARISON")
    print("="*80)
    
    comparison = trainer.compare_models()
    comparison.to_csv('model_comparison.csv', index=False)
    print("\n✓ Saved comparison to 'model_comparison.csv'")
    
    # Step 6: Feature importance
    print("\n🔍 STEP 6: FEATURE IMPORTANCE")
    print("="*80)
    
    importance = trainer.get_feature_importance('Random Forest')
    if importance is not None:
        importance.to_csv('feature_importance.csv', index=False)
        print("✓ Saved feature importance to 'feature_importance.csv'")
    
    # Step 7: Save best model
    print("\n💾 STEP 7: SAVE MODELS")
    print("="*80)
    
    trainer.save_model('Random Forest', 'random_forest_model.pkl')
    trainer.save_model('Gradient Boosting', 'gradient_boosting_model.pkl')
    trainer.save_model('Logistic Regression', 'logistic_regression_model.pkl')
    trainer.save_scaler('scaler.pkl')
    
    # Step 8: Demo prediction
    print("\n🧪 STEP 8: DEMO PREDICTIONS")
    print("="*80)
    
    predictor = FraudPredictor()
    predictor.load_model('random_forest_model.pkl')
    
    # Suspicious claim
    suspicious_claim = {
        'DAYS_TO_DECLARE': 45,
        'SAME_DAY_DECLARATION': 0,
        'LATE_DECLARATION': 1,
        'VERY_LATE_DECLARATION': 0,
        'ACCIDENT_MONTH': 3,
        'ACCIDENT_YEAR': 2015,
        'ACCIDENT_DAY_OF_WEEK': 5,
        'IS_WEEKEND': 1,
        'RESPONSABILITE': 0,
        'ZERO_RESPONSIBILITY': 1,
        'FULL_RESPONSIBILITY': 0,
        'VAGUE_LOCATION': 1,
        'HIGH_RISK_LOCATION': 0,
        'IS_MATERIAL_DAMAGE': 1,
        'MISSING_GARAGE': 1,
        'HAS_CONSTAT': 0,
        'EXPERT_FREQUENCY': 85,
        'HIGH_FREQUENCY_EXPERT': 1,
        'GARAGE_FREQUENCY': 0
    }
    
    print("\nExample 1: SUSPICIOUS CLAIM")
    result1 = predictor.explain_prediction(suspicious_claim)
    print(f"  Prediction: {result1['prediction']['prediction']}")
    print(f"  Probability: {result1['prediction']['fraud_probability']:.2f}%")
    print(f"  Risk Level: {result1['prediction']['risk_level']}")
    print("  Risk Factors:")
    for factor in result1['risk_factors']:
        print(f"    {factor}")
    
    # Normal claim
    normal_claim = {
        'DAYS_TO_DECLARE': 3,
        'SAME_DAY_DECLARATION': 0,
        'LATE_DECLARATION': 0,
        'VERY_LATE_DECLARATION': 0,
        'ACCIDENT_MONTH': 6,
        'ACCIDENT_YEAR': 2015,
        'ACCIDENT_DAY_OF_WEEK': 2,
        'IS_WEEKEND': 0,
        'RESPONSABILITE': 50,
        'ZERO_RESPONSIBILITY': 0,
        'FULL_RESPONSIBILITY': 0,
        'VAGUE_LOCATION': 0,
        'HIGH_RISK_LOCATION': 0,
        'IS_MATERIAL_DAMAGE': 1,
        'MISSING_GARAGE': 0,
        'HAS_CONSTAT': 1,
        'EXPERT_FREQUENCY': 25,
        'HIGH_FREQUENCY_EXPERT': 0,
        'GARAGE_FREQUENCY': 2
    }
    
    print("\nExample 2: NORMAL CLAIM")
    result2 = predictor.explain_prediction(normal_claim)
    print(f"  Prediction: {result2['prediction']['prediction']}")
    print(f"  Probability: {result2['prediction']['fraud_probability']:.2f}%")
    print(f"  Risk Level: {result2['prediction']['risk_level']}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PROJECT COMPLETE!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("  - preprocessed_claims.csv (processed data)")
    print("  - model_comparison.csv (model metrics)")
    print("  - feature_importance.csv (feature rankings)")
    print("  - random_forest_model.pkl (trained model)")
    print("  - gradient_boosting_model.pkl (trained model)")
    print("  - logistic_regression_model.pkl (trained model)")
    print("  - scaler.pkl (feature scaler)")
    
    print("\n🎯 Next Steps:")
    print("  1. Review model_comparison.csv for best model")
    print("  2. Check feature_importance.csv for key fraud indicators")
    print("  3. Use fraud_predictor.py for real-time predictions")
    print("  4. Present results with visualizations")
    
    print("\n🏆 Project by: Dridi Ahmed Omar: Esprit AI Student - 3rd Year")
    print("="*80)


if __name__ == "__main__":
    main()
