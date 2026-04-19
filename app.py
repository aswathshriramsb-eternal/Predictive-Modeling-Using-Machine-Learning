import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Page config
st.set_page_config(page_title="Iris Dashboard", layout="centered")

# Title
st.title("🌸 Iris Flower Prediction Dashboard")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.success(f"Accuracy: {round(acc * 100, 2)}%")

# User Input
st.subheader("🎚️ Enter Flower Measurements")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Prediction
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)

    result = iris.target_names[prediction[0]].capitalize()

    st.subheader("🌼 Prediction Result")
    st.success(f"🌸 {result}")

# Visualization
st.subheader("📈 Data Visualization")

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

fig1 = plt.figure()
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Width")

st.pyplot(fig1)

# Confusion Matrix
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig2, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

st.pyplot(fig2)