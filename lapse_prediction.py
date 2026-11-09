
from pathlib import Path

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# 1. Paths and cutoff
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


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

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["status"] = orders["status"].astype("string").str.lower()

# Only completed orders count as customer activity.
completed = orders.loc[
    (orders["status"] == "completed")
    & orders["order_ts"].notna()
    & orders["customer_id"].notna()
].copy()


# --------------------------------------------------
# 3. Define the eligible cohort
# --------------------------------------------------
# A customer qualifies if they completed at least one
# order strictly before the cutoff.
historical_completed = completed.loc[
    completed["order_ts"] < CUTOFF
].copy()

cohort_ids = historical_completed["customer_id"].unique()

cohort = customers.loc[
    customers["customer_id"].isin(cohort_ids)
].copy()

# Label from completed orders during September only.
september_completed = completed.loc[
    (completed["order_ts"] >= CUTOFF)
    & (completed["order_ts"] < LABEL_END)
]

active_in_september = set(
    september_completed["customer_id"].unique()
)

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_in_september)
).astype(int)


# --------------------------------------------------
# 4. Report and verify cohort counts
# --------------------------------------------------
print("=" * 65)
print("LAPSE COHORT AND LABEL CHECK")
print("=" * 65)

print(f"Eligible customers: {len(cohort)}")
print(f"Lapsed customers: {int(cohort['lapsed'].sum())}")
print(f"Not-lapsed customers: {int((cohort['lapsed'] == 0).sum())}")
print(f"Lapsed percentage: {cohort['lapsed'].mean() * 100:.2f}%")

print("\nLabel definition:")
print("1 = no completed order from 2025-09-01 through 2025-09-30")
print("0 = at least one completed order during that period")

assert cohort["customer_id"].is_unique, (
    "The cohort must contain one row per customer."
)
assert set(cohort["lapsed"].unique()).issubset({0, 1}), (
    "Labels must be binary."
)

print("\nCohort uniqueness and binary-label checks passed.")


# --------------------------------------------------
# 5. Build features using pre-cutoff data only
# --------------------------------------------------

# No order on or after the cutoff may enter the features.
historical_orders = orders.loc[
    (orders["order_ts"] < CUTOFF)
    & orders["status"].isin(["completed", "cancelled"])
].copy()

# Add product category to orders for favourite-category calculation.
products = pd.read_csv(DATA_DIR / "products.csv")

historical_orders = historical_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

historical_completed = historical_orders.loc[
    historical_orders["status"] == "completed"
].copy()

# Feature 1-3: recency, frequency and monetary.
rfm_features = (
    historical_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm_features["recency"] = (
    CUTOFF - rfm_features["last_order_ts"]
).dt.days

rfm_features = rfm_features[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Feature 4: average discount on completed orders.
discount_feature = (
    historical_completed.groupby("customer_id")["discount_pct"]
    .mean()
    .rename("average_discount_pct")
)

# Feature 5: cancellation rate among final-status orders.
final_order_counts = (
    historical_orders.groupby("customer_id")
    .agg(
        final_order_count=("order_id", "nunique"),
        cancelled_count=(
            "status",
            lambda values: (values == "cancelled").sum(),
        ),
    )
)

final_order_counts["cancellation_rate"] = (
    final_order_counts["cancelled_count"]
    / final_order_counts["final_order_count"]
)

# Feature 6: average completed-order delivery time.
delivery_feature = (
    historical_completed.groupby("customer_id")["delivery_minutes"]
    .mean()
    .rename("average_delivery_minutes")
)

# Feature 7: most frequently ordered product category.
def most_frequent_category(values):
    modes = values.dropna().mode()
    if modes.empty:
        return np.nan
    return sorted(modes.astype(str))[0]


# Import NumPy for the missing-value marker used above.
import numpy as np

category_feature = (
    historical_completed.groupby("customer_id")["category"]
    .agg(most_frequent_category)
    .rename("favourite_category")
)

# Combine the behaviour features.
features = (
    rfm_features
    .merge(discount_feature, on="customer_id", how="left")
    .merge(
        final_order_counts[["cancellation_rate"]],
        on="customer_id",
        how="left",
    )
    .merge(delivery_feature, on="customer_id", how="left")
    .merge(category_feature, on="customer_id", how="left")
)

# Features 8-11: city, age, gender and tenure.
customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_features["signup_date"] = pd.to_datetime(
    customer_features["signup_date"], errors="coerce"
)

customer_features["tenure_days"] = (
    CUTOFF - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(columns=["signup_date"])

features = features.merge(
    customer_features,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

# Join the target label only after building historical features.
model_data = features.merge(
    cohort[["customer_id", "lapsed"]],
    on="customer_id",
    how="inner",
    validate="one_to_one",
)

# Confirm that all feature orders precede the cutoff.
assert historical_orders["order_ts"].max() < CUTOFF
assert historical_completed["order_ts"].max() < CUTOFF
assert len(model_data) == len(cohort)
assert model_data["customer_id"].is_unique

print("\n" + "=" * 65)
print("PRE-CUTOFF FEATURE CHECK")
print("=" * 65)
print(f"Model rows: {len(model_data)}")
print(f"Feature columns: {len(model_data.columns) - 2}")
print("Features:")
print(", ".join(
    column for column in model_data.columns
    if column not in ["customer_id", "lapsed"]
))
print(f"Latest order used in features: {historical_orders['order_ts'].max()}")
print(f"Cutoff: {CUTOFF}")
print("\nMissing values per column:")
print(model_data.isna().sum().to_string())


# --------------------------------------------------
# 6. Train/test split and baseline evaluation
# --------------------------------------------------

# Exclude customer_id and the target from the feature matrix.
X = model_data.drop(columns=["customer_id", "lapsed"]).copy()
y = model_data["lapsed"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\n" + "=" * 65)
print("TRAIN/TEST SPLIT")
print("=" * 65)
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Training lapsed: {int(y_train.sum())}")
print(f"Test lapsed: {int(y_test.sum())}")

assert len(X_train) == 924
assert len(X_test) == 232

# Baseline: always predict not lapsed (class 0).
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\n" + "=" * 65)
print("BASELINE: ALWAYS PREDICT NOT LAPSED")
print("=" * 65)
print(f"Accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
print(
    "Lapsed-class precision:",
    f"{precision_score(y_test, baseline_predictions, zero_division=0):.4f}",
)
print(
    "Lapsed-class recall:",
    f"{recall_score(y_test, baseline_predictions, zero_division=0):.4f}",
)
print(
    "Lapsed-class F1:",
    f"{f1_score(y_test, baseline_predictions, zero_division=0):.4f}",
)
print("Confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))


# --------------------------------------------------
# 7. Leakage-safe preprocessing
# --------------------------------------------------

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]

categorical_features = [
    "city",
    "gender",
    "favourite_category",
]

# Numeric preprocessing: impute and standardise.
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Categorical preprocessing: impute and one-hot encode.
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# Fit on training rows only, then transform the test rows.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Confirm that preprocessing produced no missing or infinite values.
train_array = np.asarray(X_train_processed, dtype=float)
test_array = np.asarray(X_test_processed, dtype=float)

assert np.isfinite(train_array).all(), (
    "Training features contain missing or infinite values."
)
assert np.isfinite(test_array).all(), (
    "Test features contain missing or infinite values."
)

# The frequency scaler's learned mean comes from training rows only.
frequency_mean_train = X_train["frequency"].mean()
frequency_mean_full = X["frequency"].mean()

learned_frequency_mean = (
    preprocessor
    .named_transformers_["numeric"]
    .named_steps["scaler"]
    .mean_[numeric_features.index("frequency")]
)

print("\n" + "=" * 65)
print("PREPROCESSING VALIDATION")
print("=" * 65)
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Processed training shape: {train_array.shape}")
print(f"Processed test shape: {test_array.shape}")
print(f"Learned training frequency mean: {learned_frequency_mean:.4f}")
print(f"Full-cohort frequency mean: {frequency_mean_full:.4f}")
print(f"Training-only frequency mean: {frequency_mean_train:.4f}")

assert np.isclose(learned_frequency_mean, frequency_mean_train)

print(
    "\nThe scaler mean differs from the full-cohort mean because "
    "the scaler is fitted only on training rows, not on the test rows."
)
print("No missing or infinite values remain after preprocessing.")


# --------------------------------------------------
# 8. Tune models using training data only
# --------------------------------------------------

# Each model gets its own pipeline so preprocessing is
# refitted within every cross-validation training fold.
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

print("\n" + "=" * 65)
print("TUNING LOGISTIC REGRESSION")
print("=" * 65)

logistic_grid.fit(X_train, y_train)

best_logistic = logistic_grid.best_estimator_

print("Best parameters:", logistic_grid.best_params_)
print(f"Best cross-validated F1: {logistic_grid.best_score_:.4f}")


# Random forest uses the same leakage-safe preprocessing.
forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

forest_grid = GridSearchCV(
    estimator=forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

print("\n" + "=" * 65)
print("TUNING RANDOM FOREST")
print("=" * 65)

forest_grid.fit(X_train, y_train)

best_forest = forest_grid.best_estimator_

print("Best parameters:", forest_grid.best_params_)
print(f"Best cross-validated F1: {forest_grid.best_score_:.4f}")


# --------------------------------------------------
# 9. Evaluate both tuned models on the untouched test set
# --------------------------------------------------

def report_test_metrics(model_name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print(f"\n{model_name}")
    print("-" * 45)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_eval, predictions, labels=[0, 1]))
    print(
        f"Lapsed-class precision: "
        f"{precision_score(y_eval, predictions, pos_label=1, zero_division=0):.4f}"
    )
    print(
        f"Lapsed-class recall: "
        f"{recall_score(y_eval, predictions, pos_label=1, zero_division=0):.4f}"
    )
    print(
        f"Lapsed-class F1: "
        f"{f1_score(y_eval, predictions, pos_label=1, zero_division=0):.4f}"
    )

    return predictions


print("\n" + "=" * 65)
print("FINAL TEST-SET EVALUATION")
print("=" * 65)

logistic_test_predictions = report_test_metrics(
    "Tuned Logistic Regression",
    best_logistic,
    X_test,
    y_test,
)

forest_test_predictions = report_test_metrics(
    "Tuned Random Forest",
    best_forest,
    X_test,
    y_test,
)


# --------------------------------------------------
# 10. Oversample the lapsed class in training data only
# --------------------------------------------------

from sklearn.base import clone
from sklearn.utils import resample

# Keep every original training row.
balanced_X = X_train.copy()
balanced_y = y_train.copy()

# Separate the minority class from training data only.
lapsed_X = X_train.loc[y_train == 1]
lapsed_y = y_train.loc[y_train == 1]

non_lapsed_count = int((y_train == 0).sum())
lapsed_count = int((y_train == 1).sum())
duplicates_needed = non_lapsed_count - lapsed_count

if duplicates_needed > 0:
    extra_X, extra_y = resample(
        lapsed_X,
        lapsed_y,
        replace=True,
        n_samples=duplicates_needed,
        random_state=42,
    )

    balanced_X = pd.concat(
        [balanced_X, extra_X],
        axis=0,
    )
    balanced_y = pd.concat(
        [balanced_y, extra_y],
        axis=0,
    )

print("\n" + "=" * 65)
print("CLASS-BALANCE EXPERIMENT")
print("=" * 65)
print("Original training class counts:")
print(y_train.value_counts().sort_index().to_dict())
print("Balanced training class counts:")
print(balanced_y.value_counts().sort_index().to_dict())

assert balanced_y.value_counts()[0] == balanced_y.value_counts()[1]
assert len(balanced_X) == len(balanced_y)

# Clone the best tuned logistic pipeline.
# Its preprocessing will be fitted again on balanced training rows only.
balanced_logistic = clone(best_logistic)
balanced_logistic.fit(balanced_X, balanced_y)

# Evaluate on the unchanged, original test set.
balanced_predictions = balanced_logistic.predict(X_test)

print("\nLogistic regression comparison on the same test set:")
print(
    f"Original precision: "
    f"{precision_score(y_test, logistic_test_predictions, zero_division=0):.4f}"
)
print(
    f"Original recall: "
    f"{recall_score(y_test, logistic_test_predictions, zero_division=0):.4f}"
)
print(
    f"Original F1: "
    f"{f1_score(y_test, logistic_test_predictions, zero_division=0):.4f}"
)
print(
    f"Balanced precision: "
    f"{precision_score(y_test, balanced_predictions, zero_division=0):.4f}"
)
print(
    f"Balanced recall: "
    f"{recall_score(y_test, balanced_predictions, zero_division=0):.4f}"
)
print(
    f"Balanced F1: "
    f"{f1_score(y_test, balanced_predictions, zero_division=0):.4f}"
)


# --------------------------------------------------
# 11. Interpret models, save the selected model,
#     and create the Premium retention list
# --------------------------------------------------

import joblib

# Select the random forest because it had the higher
# test F1 score than logistic regression.
selected_model = best_forest

# --------------------------------------------------
# A. Five most important random forest features
# --------------------------------------------------
forest_preprocessor = best_forest.named_steps["preprocessor"]
forest_model = best_forest.named_steps["model"]

forest_feature_names = forest_preprocessor.get_feature_names_out()
forest_importances = forest_model.feature_importances_

importance_table = pd.DataFrame(
    {
        "feature": forest_feature_names,
        "importance": forest_importances,
    }
).sort_values("importance", ascending=False)

print("\n" + "=" * 65)
print("TOP FIVE RANDOM FOREST FEATURES")
print("=" * 65)
print(importance_table.head(5).to_string(index=False))

print("\nBusiness interpretation:")
print(
    "Higher-importance features contributed more to the forest's "
    "prediction decisions; importance does not establish causation."
)

# --------------------------------------------------
# B. Signs of three logistic regression coefficients
# --------------------------------------------------
logistic_preprocessor = best_logistic.named_steps["preprocessor"]
logistic_model = best_logistic.named_steps["model"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()
logistic_coefficients = logistic_model.coef_[0]

coefficient_table = pd.DataFrame(
    {
        "feature": logistic_feature_names,
        "coefficient": logistic_coefficients,
    }
)

# Prefer three meaningful numeric behaviour features.
chosen_features = [
    "numeric__recency",
    "numeric__frequency",
    "numeric__monetary",
]

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION COEFFICIENT SIGNS")
print("=" * 65)

for feature in chosen_features:
    match = coefficient_table.loc[
        coefficient_table["feature"] == feature
    ]

    if match.empty:
        print(f"{feature}: not found in transformed features.")
        continue

    coefficient = float(match["coefficient"].iloc[0])
    sign = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"

    print(f"{feature}: coefficient={coefficient:.4f} ({sign})")

    if feature == "numeric__recency":
        meaning = (
            "A positive sign means greater time since the last order "
            "is associated with higher predicted lapse odds, holding "
            "other encoded features constant."
        )
    elif feature == "numeric__frequency":
        meaning = (
            "A negative sign means more completed orders are associated "
            "with lower predicted lapse odds, holding other features constant."
        )
    else:
        if coefficient > 0:
            meaning = (
                "The positive coefficient means higher monetary value "
                "is associated with higher predicted lapse odds, "
                "holding other features constant."
            )
        elif coefficient < 0:
            meaning = (
                "The negative coefficient means higher monetary value "
                "is associated with lower predicted lapse odds, "
                "holding other features constant."
            )
        else:
            meaning = (
                "The zero coefficient indicates no linear association "
                "with predicted lapse odds for this feature, "
                "holding other features constant."
            )   

    print(f"  Business reading: {meaning}")

# --------------------------------------------------
# C. Save and reload the selected model
# --------------------------------------------------
model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(selected_model, model_path)
reloaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions,
)

print("\n" + "=" * 65)
print("MODEL PERSISTENCE CHECK")
print("=" * 65)
print(f"Saved model: {model_path}")
print(f"Reloaded predictions match exactly: {predictions_match}")

assert predictions_match, (
    "Reloaded model predictions differ from original predictions."
)

# --------------------------------------------------
# D. Create the Premium retention list from test rows
# --------------------------------------------------
segments_path = OUTPUT_DIR / "customer_segments.csv"
segments = pd.read_csv(segments_path)

# Ensure the Premium flag is interpreted as a Boolean.
if segments["is_premium"].dtype != bool:
    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

if segments["is_premium"].isna().any():
    raise ValueError("Could not interpret all Premium flags.")

# Preserve the test-row indices to map predictions back to customers.
test_customers = pd.DataFrame(
    {
        "customer_id": model_data.loc[X_test.index, "customer_id"].values,
        "lapse_probability": selected_model.predict_proba(X_test)[:, 1],
        "actually_lapsed": y_test.values,
    },
    index=X_test.index,
)

test_customers = test_customers.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left",
    validate="one_to_one",
)

premium_test = test_customers.loc[
    test_customers["is_premium"] == True
].copy()

retention_list = (
    premium_test
    .sort_values("lapse_probability", ascending=False)
    .head(10)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
    .reset_index(drop=True)
)

retention_path = OUTPUT_DIR / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

true_lapses_in_list = int(retention_list["actually_lapsed"].sum())

print("\n" + "=" * 65)
print("PREMIUM RETENTION LIST")
print("=" * 65)
print(retention_list.to_string(index=False))
print(f"\nPremium customers in test set: {len(premium_test)}")
print(f"Customers written to retention list: {len(retention_list)}")
print(f"Actually lapsed in retention list: {true_lapses_in_list}")
print(f"Saved retention list: {retention_path}")

print(
    "\nLimitation: The Premium flag is derived from whole-window RFM "
    "that includes September activity, so this retention list is "
    "illustrative and is not a leakage-free evaluation."
)

assert len(retention_list) <= 10
assert list(retention_list.columns) == [
    "customer_id",
    "lapse_probability",
    "actually_lapsed",
]
assert retention_list["lapse_probability"].is_monotonic_decreasing

print("\nPart B model interpretation and persistence checks completed.")

