import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib

file_path = 'last_tiktok.csv'  # Replace with your dataset path
tiktok_data = pd.read_csv(file_path)

tiktok_data['Upload Date'] = pd.to_datetime(tiktok_data['Upload Date'])

def time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

tiktok_data['Time of Day'] = tiktok_data['Upload Date'].dt.hour.apply(time_of_day)

tiktok_data['Likes-to-Views Ratio'] = (tiktok_data['Likes'] / tiktok_data['Video Views']).replace(
    [float('inf'), -float('inf')], 0).fillna(0)
tiktok_data['Engagement Level'] = pd.cut(
    tiktok_data['Likes-to-Views Ratio'],
    bins=[-float('inf'), 0.02, 0.05, float('inf')],
    labels=['Low', 'Medium', 'High']
)

vectorizer = CountVectorizer()
hashtags_vectorized = vectorizer.fit_transform(tiktok_data['Description'].fillna(""))

encoder = OneHotEncoder(sparse_output=False)
time_of_day_encoded = encoder.fit_transform(tiktok_data[['Time of Day']])
time_of_day_df = pd.DataFrame(time_of_day_encoded, columns=encoder.get_feature_names_out(['Time of Day']))

features = pd.DataFrame(hashtags_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
features['Log Followers'] = np.log1p(tiktok_data['User Followers'])
features['Duration (seconds)'] = tiktok_data['Duration (seconds)']

features = pd.concat([features, time_of_day_df], axis=1)

target = tiktok_data['Engagement Level']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, target)

scaler = StandardScaler()
numerical_cols = ['Log Followers', 'Duration (seconds)']
X_resampled[numerical_cols] = scaler.fit_transform(X_resampled[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

joblib.dump(rf_classifier, 'random_forest_model.pkl')  # Save the model
joblib.dump(encoder, 'time_of_day_encoder.pkl')  # Save the encoder
joblib.dump(vectorizer, 'hashtags_vectorizer.pkl')  # Save the vectorizer
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

print("Model and preprocessing tools saved!")
