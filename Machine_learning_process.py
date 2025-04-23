import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# --------------------------------
# 1. data preparation
# --------------------------------
def data_prep(filepath):
    df = pd.read_csv(filepath)
    
    # Sayısal sütunları standardize et
    num_cols = [col for col in df.columns if df[col].dtypes != "O" and col != "disease"]
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    
    # Hedef değişkeni encode et
    le = LabelEncoder()
    df['disease'] = le.fit_transform(df['disease'])

    X = df.drop(["disease"], axis=1)
    y = df["disease"]

    return X, y, le  # LabelEncoder 

# --------------------------------
# 2. split data and apply SMOTE
# --------------------------------
def split_and_smote(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        random_state=random_state)
    
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_test, y_train_smote, y_test

# --------------------------------
# 3. Base models performance
# --------------------------------
def base_models(X, y, scoring="roc_auc_ovr"):
    print("Base Models....")
    classifiers = [
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier(class_weight='balanced')),
        ('Adaboost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ]

    for name, classifier in classifiers:
        try:
            cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
        except Exception as e:
            print(f"{name} - Hata: {str(e)}")

# --------------------------------
# 4. Hyperparameter optimization 
# --------------------------------
def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
    print("Hyperparameter Optimization....")
    rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}

    classifiers = [("RF", RandomForestClassifier(), rf_params)]
    best_models = {}

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# --------------------------------
# 6. MODEL TEST DEĞERLENDİRME
# --------------------------------
def evaluate_on_test(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

# --------------------------------
# 7. Main function
# --------------------------------
if __name__ == "__main__":
    X, y, le = data_prep("/Users/beyzacanakci/Desktop/miuul/proje/final_df.csv")
    print("Class distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = split_and_smote(X, y)
    
    base_models(X_train, y_train)
    
    best_models = hyperparameter_optimization(X_train, y_train)

    # Random forest
    rf_model = best_models["RF"]
    rf_model.fit(X_train, y_train)

    evaluate_on_test(rf_model, X_test, y_test, le)
    
    importances = rf_model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    print("\nFeature Importances:\n")
    print(importance_df)

    # prediction example and model saving 
    random_user = X.sample(1, random_state=19)
    print("Tahmin edilen sınıf:", le.inverse_transform(rf_model.predict(random_user)))
    joblib.dump(rf_model, "/Users/beyzacanakci/Desktop/miuul/proje/RF-Metagenomic-data-analysis.pkl")

label_map = {i: label for i, label in enumerate(le.classes_)}
print(label_map)
