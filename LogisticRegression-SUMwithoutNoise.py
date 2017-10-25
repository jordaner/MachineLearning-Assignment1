from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["target_class"] = encoder.fit_transform(df["Target Class"])