import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Annual_Income": [15, 18, 20, 25, 30, 60, 65, 70, 80, 85],
    "Spending_Score": [80, 75, 85, 70, 65, 40, 45, 35, 30, 25]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
X = scaler.fit_transform(df)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

print("Customer Segmentation Results:")
print(df)

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.savefig("customer_segments.png")
plt.show()