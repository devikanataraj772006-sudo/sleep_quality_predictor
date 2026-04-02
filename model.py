import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:/Users/ADMIN/Downloads/sleep_data.csv")

# Clean data
df = df.dropna()
df.columns = df.columns.str.strip()

# Encode categorical columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['BMI Category'] = le.fit_transform(df['BMI Category'])

# Target (convert to 3 classes)
y = df['Quality of Sleep']
y = pd.cut(y, bins=[0,4,7,10], labels=[0,1,2])

# Features
X = df[['Sleep Duration',
        'Physical Activity Level',
        'Stress Level',
        'Heart Rate',
        'Daily Steps',
        'Gender',
        'BMI Category']]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model trained successfully!")
