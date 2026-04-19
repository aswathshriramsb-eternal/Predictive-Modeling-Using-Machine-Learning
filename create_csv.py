import pandas as pd
from sklearn.datasets import load_iris

print("Creating iris.csv...")

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

df.to_csv("iris.csv", index=False)

print("✅ iris.csv created successfully!")