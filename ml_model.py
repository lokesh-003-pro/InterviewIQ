import os
from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'dataset'/'training_products.csv'
MODEL=ROOT/'model'/'product_models.joblib'
FEATURES=['rating','review_count','positive_review_pct','negative_review_pct','seller_rating','price','discount_pct','delivery_issue_pct']

def train_models():
    df=pd.read_csv(DATA)
    X=df[FEATURES]
    y=df['quality_label']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=250,random_state=42,class_weight='balanced'))
    clf.fit(Xtr,ytr)
    acc=accuracy_score(yte,clf.predict(Xte))
    # Training labels are synthetic demo data; do not interpret this score as real-world validation.
    risk=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=200,random_state=42))
    risk.fit(X,df['return_risk_pct'])
    joblib.dump({'quality_model':clf,'risk_model':risk,'features':FEATURES,'demo_data':True},MODEL)
    return acc

def load_models():
    if not MODEL.exists(): train_models()
    bundle=joblib.load(MODEL)
    if not isinstance(bundle,dict) or 'quality_model' not in bundle or 'risk_model' not in bundle:
        raise RuntimeError('Saved model file is incompatible. Delete model/product_models.joblib and run python app/train_model.py')
    return bundle

def predict(features):
    bundle=load_models()
    row={k:features.get(k) for k in bundle['features']}
    X=pd.DataFrame([row],columns=bundle['features'])
    quality=str(bundle['quality_model'].predict(X)[0])
    try: risk=float(bundle['risk_model'].predict(X)[0])
    except Exception: risk=50.0
    risk=max(0,min(100,risk))
    level='Low' if risk<25 else 'Medium' if risk<55 else 'High'
    # Confidence here is the model's class probability, not a guarantee of correctness.
    probs=bundle['quality_model'].predict_proba(X)[0]
    confidence=round(float(max(probs))*100)
    return {'quality':quality,'return_risk_pct':round(risk,1),'risk_level':level,'model_confidence':confidence,'demo_model':True}
