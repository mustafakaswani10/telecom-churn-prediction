import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import joblib

# Load the data
data_path = '/Users/tafy/Desktop/Python/telecom-churn-prediction/data/telecom_churn.csv'
telecom_data = pd.read_csv(data_path)

# Feature Engineering
telecom_data['CallDuration'] = telecom_data['DayMins'] / telecom_data['DayCalls']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = telecom_data.drop('Churn', axis=1)
y = telecom_data['Churn']
X_imputed = imputer.fit_transform(X)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
gbm = GradientBoostingClassifier(random_state=42)

models = {'Logistic Regression': log_reg, 'Random Forest': rf, 'Gradient Boosting Machine': gbm}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Classification Report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {name}')
    plt.legend(loc="lower right")
    plt.show()

    # Save the model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
