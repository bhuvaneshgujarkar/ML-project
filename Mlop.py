# 3
# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Step 2: Load the dataset
path = r"C:\Users\bhuva\OneDrive\Desktop\Twitter_Data.csv"
df = pd.read_csv(path)

# Display first few rows
print(df.head())

# Step 3: Preprocessing (Assuming 'text' and 'label' columns)
df.dropna(inplace=True)
X = df['text']  # Replace with the actual column name for text
y = df['label']  # Replace with the actual column name for labels

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build Pipeline with TF-IDF and Logistic Regression
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr', LogisticRegression())
])

# Step 6: Train the model
model_pipeline.fit(X_train, y_train)

# Step 7: Save the model
joblib.dump(model_pipeline, 'twitter_sentiment_model.pkl')
print("✅ Model saved as twitter_sentiment_model.pkl")

# Optional: Evaluate on test set
print("Test Accuracy:", model_pipeline.score(X_test, y_test))


# reuse model
# Step 8: Load the saved model
loaded_model = joblib.load('twitter_sentiment_model.pkl')

# Step 9: Make predictions
sample_tweets = [
    "I love this product! It's amazing.",
    "Worst service ever. Totally disappointed!"
]

predictions = loaded_model.predict(sample_tweets)
print(predictions)






# 4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
df = pd.read_csv(r"C:\Users\bhuva\OneDrive\Desktop\Twitter_Data.csv")
df.head()
df.dropna(inplace=True)
X = df['text']  # Adjust if your column name is different
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('lr', LogisticRegression())
])
model.fit(X_train, y_train)
print("Test Accuracy:", model.score(X_test, y_test))
import numpy as np
labels = ['Negative', 'Positive']
plt.bar(labels, np.bincount(model.predict(X_test)))
plt.title("Prediction Distribution")
plt.show()
joblib.dump(model, 'twitter_model.pkl')



# 5
# titanic_eda.ipynb (write in Jupyter)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")
df.head()

# Null values
print("Missing values:\n", df.isnull().sum())

# Fill or drop nulls
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Basic summary
print(df.describe())
print(df.info())

# Visualizations
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.show()

sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

sns.boxplot(data=df, x='Pclass', y='Age')
plt.title('Age by Passenger Class')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()







# 6
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report

# Create results directory
os.makedirs("results", exist_ok=True)

# Load Dataset
df = pd.read_csv("train.csv")

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Convert categorical
df = pd.get_dummies(df, drop_first=True)

# Features and Labels
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/confusion_{name.replace(' ', '').lower()}.png")
    plt.close()

    # Precision-Recall Curve
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=name)

    # Classification Report
    print(f"\n{name} Report:")
    print(classification_report(y_test, y_pred))

# Final Precision-Recall Plot
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.savefig("results/pr_curve_comparison.png")
plt.close()

print("✅ All plots and reports saved in /results/")
