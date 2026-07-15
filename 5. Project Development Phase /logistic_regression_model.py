import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_credit_model():
    # 1. Load Dataset
    df = pd.read_csv("Application_Data.csv")

    # Feature Selection
    feature_columns = ["Total_Income", "Applicant_Age", "Total_Bad_Debt", "Total_Good_Debt"]
    X = df[feature_columns]
    y = df["Status"]

    # 🛑 PRINT THE CHEAT SHEET TO TERMINAL
    print("\n==============================================")
    print("📋 DATASET TARGET CHEAT SHEET:")
    print(df["Status"].value_counts())
    print("==============================================\n")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced_subsample',
        max_depth=12
    )
    model.fit(X_train, y_train)

    with open("credit_card_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("✅ Model successfully saved to credit_card_model.pkl!")

if __name__ == "__main__":
    train_credit_model()
