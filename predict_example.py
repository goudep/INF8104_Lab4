"""Predict one illustrative application with the frozen NorVenta teaching baseline.

Usage:
  python predict_example.py
  python predict_example.py --revenu 50000 --score-credit 700

Only load joblib files from a trusted team source.
This is a coursework example, NOT a credit decision engine.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="NorVenta teaching baseline: predict one example")
    parser.add_argument('--revenu', type=float, default=50000.0, help='Example income (same units as CSV)')
    parser.add_argument('--score-credit', type=float, default=700.0, help='Example credit score')
    args = parser.parse_args()

    model = joblib.load(BASE / 'norventa_logreg_model.joblib')
    X = pd.DataFrame([{'revenu': args.revenu, 'score_credit': args.score_credit}])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])
    print('MODEL: team-built teaching baseline (not NorVenta production model)')
    print(f'INPUT: revenu={args.revenu:.2f}; score_credit={args.score_credit:.2f}')
    print(f'PREDICTION: {prediction} (1=approval label, 0=rejection label)')
    print(f'DECISION: {"Approved" if prediction == 1 else "Not approved"}')
    print(f'MODEL P(approbation=1): {probability:.4f}')
    print('NOTE: Illustrative model prediction only. Not an actual credit approval.')

if __name__ == '__main__':
    main()
