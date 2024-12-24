import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from textblob import TextBlob
from datetime import datetime

file_path = 'last_tiktok.csv'  # Replace with your dataset path
tiktok_data = pd.read_csv(file_path)

def classify_time_period(upload_time):
    hour = upload_time.hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

tiktok_data['Upload Date'] = pd.to_datetime(tiktok_data['Upload Date'], errors='coerce')
tiktok_data['Time Period'] = tiktok_data['Upload Date'].apply(lambda x: classify_time_period(x) if pd.notnull(x) else "Unknown")

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

tiktok_data['Sentiment'] = tiktok_data['Description'].fillna("").apply(get_sentiment)
tiktok_data['Unique Hashtags'] = tiktok_data['Description'].str.findall(r"#\w+").apply(lambda x: len(set(x)) if x else 0)

tiktok_data['Likes-to-Views Ratio'] = (tiktok_data['Likes'] / tiktok_data['Video Views']).replace(
    [float('inf'), -float('inf')], 0).fillna(0)
tiktok_data['Engagement Level'] = pd.cut(
    tiktok_data['Likes-to-Views Ratio'],
    bins=[-float('inf'), 0.02, 0.05, float('inf')],
    labels=['Low', 'Medium', 'High']
)

vectorizer = CountVectorizer()
hashtags_vectorized = vectorizer.fit_transform(tiktok_data['Description'].fillna(""))

features = pd.DataFrame(hashtags_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
features['Log Followers'] = np.log1p(tiktok_data['User Followers'])
features['Duration (seconds)'] = tiktok_data['Duration (seconds)']
features['Hashtag Count'] = tiktok_data['Description'].str.count(r"#\w+").fillna(0)
features['Unique Hashtags'] = tiktok_data['Unique Hashtags']

sentiment_encoded = pd.get_dummies(tiktok_data['Sentiment'], prefix='Sentiment')
time_period_encoded = pd.get_dummies(tiktok_data['Time Period'], prefix='TimePeriod')
features = pd.concat([features, sentiment_encoded, time_period_encoded], axis=1)

target = tiktok_data['Engagement Level']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

label_encoder = LabelEncoder()
y_resampled_encoded = label_encoder.fit_transform(y_resampled)
y_test_encoded = label_encoder.transform(target)

scaler = StandardScaler()
numerical_cols = ['Log Followers', 'Duration (seconds)', 'Hashtag Count', 'Unique Hashtags']
X_resampled[numerical_cols] = scaler.fit_transform(X_resampled[numerical_cols])
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled_encoded, test_size=0.2, random_state=42)

param_grid_lgbm = {
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [-1, 5, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
    'reg_lambda': [0, 0.1, 0.5, 1.0],  # L2 regularization
}

lgbm_random = RandomizedSearchCV(
    estimator=LGBMClassifier(random_state=42, objective='multiclass', n_jobs=-1),
    param_distributions=param_grid_lgbm,
    n_iter=50,  # Number of parameter settings to sample
    scoring='accuracy',
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

lgbm_random.fit(X_train, y_train)

best_params_lgbm = lgbm_random.best_params_
print("Best Hyperparameters (LightGBM):\n", best_params_lgbm)

best_lgbm = lgbm_random.best_estimator_

y_pred_best_lgbm = best_lgbm.predict(X_test)
report_best_lgbm = classification_report(y_test, y_pred_best_lgbm, target_names=label_encoder.classes_)
conf_matrix_best_lgbm = confusion_matrix(y_test, y_pred_best_lgbm)

print("\nClassification Report (Best LightGBM):\n", report_best_lgbm)
print("\nConfusion Matrix (Best LightGBM):\n", conf_matrix_best_lgbm)
