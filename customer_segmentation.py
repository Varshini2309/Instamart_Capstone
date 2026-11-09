
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Paths and fixed settings
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4


# --------------------------------------------------
# 2. Load clean data
# --------------------------------------------------
orders = pd.read_csv(DATA_DIR / "orders.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

completed = orders.loc[
    (orders["status"].str.lower() == "completed")
    & orders["order_ts"].notna()
    & orders["customer_id"].notna()
].copy()

# The snapshot defines which completed orders belong in RFM.
completed = completed.loc[
    completed["order_ts"] < SNAPSHOT
].copy()

if completed.empty:
    raise ValueError("No completed orders are available for RFM.")

# --------------------------------------------------
# 3. Build the RFM table
# --------------------------------------------------
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (
    SNAPSHOT - rfm["last_order_ts"]
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

rfm["frequency"] = rfm["frequency"].astype(int)
rfm["recency"] = rfm["recency"].astype(int)

if rfm[["recency", "frequency", "monetary"]].isna().any().any():
    raise ValueError("RFM contains missing values.")

if (rfm[["recency", "frequency", "monetary"]] < 0).any().any():
    raise ValueError("RFM contains negative values.")

print("\n" + "=" * 65)
print("RFM SUMMARY")
print("=" * 65)
print(f"Customers with completed orders: {len(rfm)}")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# --------------------------------------------------
# 4. Standardise RFM and compare K = 2 to 8
# --------------------------------------------------
rfm_columns = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm[rfm_columns])

cluster_results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )
    labels = kmeans.fit_predict(scaled_rfm)

    cluster_results.append(
        {
            "K": k,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(scaled_rfm, labels),
        }
    )

cluster_table = pd.DataFrame(cluster_results)

print("\n" + "=" * 65)
print("K-MEANS COMPARISON")
print("=" * 65)
print(cluster_table.round(4).to_string(index=False))

# --------------------------------------------------
# 5. Fit the selected K and profile the segments
# --------------------------------------------------
final_model = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42,
)

rfm["segment_id"] = final_model.fit_predict(scaled_rfm)

profiles = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .reset_index()
)

# Cluster IDs are arbitrary labels, not customer rankings.
premium_segment = int(
    profiles.loc[profiles["mean_monetary"].idxmax(), "segment_id"]
)

# Give each cluster a descriptive name based on its profile.

def name_segment(row):
    if row["segment_id"] == premium_segment:
        return "Premium"

    if row["mean_recency"] >= 60:
        return "Dormant Customers"

    if (
        row["mean_frequency"] >= profiles["mean_frequency"].median()
        and row["mean_recency"] <= profiles["mean_recency"].median()
    ):
        return "Loyal and Active"

    return "Occasional Customers"



name_lookup = {
    int(row["segment_id"]): name_segment(row)
    for _, row in profiles.iterrows()
}

rfm["segment_name"] = rfm["segment_id"].map(name_lookup)
rfm["is_premium"] = rfm["segment_id"].eq(premium_segment)

print("\n" + "=" * 65)
print(f"SEGMENT PROFILES (K={FINAL_K})")
print("=" * 65)
print(profiles.round(2).to_string(index=False))
print(f"\nPremium segment ID: {premium_segment}")
print("Segment IDs are arbitrary labels and have no inherent ordering.")

print("\nRecommended campaign actions:")
for _, row in profiles.iterrows():
    segment_id = int(row["segment_id"])
    name = name_lookup[segment_id]

    if segment_id == premium_segment:
        action = "Offer VIP benefits and personalised loyalty rewards."
    elif name == "Loyal and Active":
        action = "Encourage repeat purchases with relevant cross-sell offers."
    elif name == "At-Risk / Inactive":
        action = "Send a targeted win-back campaign and monitor response."
    else:
        action = "Use relevant recommendations and moderate-value incentives."

    print(f"- {name}: {action}")

# --------------------------------------------------
# 6. PCA on the standardised RFM matrix
# --------------------------------------------------
pca = PCA(n_components=2)
pca.fit(scaled_rfm)

explained = pca.explained_variance_ratio_

print("\n" + "=" * 65)
print("PCA RESULTS")
print("=" * 65)
print(f"PC1 explained variance: {explained[0]:.4f}")
print(f"PC2 explained variance: {explained[1]:.4f}")
print(f"Total explained variance: {explained.sum():.4f}")

# --------------------------------------------------
# 7. Save the required segment file
# --------------------------------------------------
segment_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].copy()

segment_output["is_premium"] = (
    segment_output["is_premium"].astype(bool)
)

output_path = OUTPUT_DIR / "customer_segments.csv"
segment_output.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("SEGMENT FILE SAVED")
print("=" * 65)
print(f"File: {output_path}")
print(f"Rows: {len(segment_output)}")
print(f"Premium customers: {segment_output['is_premium'].sum()}")
print("Columns:", ", ".join(segment_output.columns))
