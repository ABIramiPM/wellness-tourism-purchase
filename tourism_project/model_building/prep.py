import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

#Customer ID is only an identifier so it will be droped.
df.drop(columns=["CustomerID"], inplace=True)


X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratify=y keeps the (imbalaced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y , test_size=0.2, random_state=42, stratify=y
)


Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


print("Data prepared: train/test splits written.")
print("Categorical values kept as:")
print("TypeofContact:", sorted(X["TypeofContact"].dropna().unique()))
print("Occupation:", sorted(X["Occupation"].dropna().unique()))
print("Gender:", sorted(X["Gender"].dropna().unique()))
print("MaritalStatus:", sorted(X["MaritalStatus"].dropna().unique()))
print("Designation:", sorted(X["Designation"].dropna().unique()))
print("ProductPitched:", sorted(X["ProductPitched"].dropna().unique()))
