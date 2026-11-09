# Instamart Customer Intelligence & Support Copilot

## Overview

The Instamart Customer Intelligence & Support Copilot is a data analytics and customer support project built using Python, pandas, SQLite, SQL, and an offline mock LLM.

The project uses customer, product, order, and support-ticket data to build a reproducible data foundation for customer analysis, retention modelling, support-ticket classification, and policy-based refund decisions.

The project is divided into four parts:

- **Part A — Data Foundation:** Clean raw datasets, store the clean tables in SQLite, validate the data, and perform SQL and pandas analysis.
- **Part B — Customer Intelligence:** Analyse customer behaviour and build a customer retention model.
- **Part C — Support Ticket Classification:** Process and classify support tickets with appropriate validation and privacy safeguards.
- **Part D — Support Copilot:** Implement policy-based support workflows using the supplied offline mock LLM.

## Reproduce

Run the following commands from the repository root in PowerShell on Windows.

1. Create a Python 3.11 virtual environment:

   ```powershell
   py -3.11 -m venv .venv311
   ```

2. Install the required dependencies:

   ```powershell
   .\.venv311\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Generate the raw datasets and policy documents:

   ```powershell
   .\.venv311\Scripts\python.exe generate_data.py
   ```

4. Audit the raw data:

   ```powershell
   .\.venv311\Scripts\python.exe audit_data.py
   ```

5. Clean the data and create the SQLite database:

   ```powershell
   .\.venv311\Scripts\python.exe clean_data.py
   ```

6. Validate the clean datasets and database:

   ```powershell
   .\.venv311\Scripts\python.exe validate_data.py
   ```

7. Execute the seven SQL queries:

   ```powershell
   .\.venv311\Scripts\python.exe run_sql_analysis.py
   ```

8. Run the pandas analysis:

   ```powershell
   .\.venv311\Scripts\python.exe pandas_analysis.py
   ```
9. Run customer segmentation and K-means analysis:

   ```powershell
   .\.venv311\Scripts\python.exe customer_segmentation.py
   ```

10. Train, tune, and evaluate the customer lapse prediction models:

    ```powershell
    .\.venv311\Scripts\python.exe lapse_prediction.py
    ```
    
## Part A — Data Foundation: Clean, Store, Query

### Raw Data Audit

The raw datasets were generated using the supplied `generate_data.py` script. The audit was performed with pandas before cleaning, and the raw CSV files were preserved as the source of truth.

| Audit checkpoint | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Nonnumeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank customer genders | 30 |

The audit script also reports the shape, data types, and missing-value counts and percentages for the raw tables.

### Customer Cleaning

The customer cleaning rules were applied as follows:

1. **City standardization:** Leading and trailing spaces were removed, and city names were converted to title case. The 24 raw spellings were standardized into six cities.
2. **Gender standardization:** Gender values were converted to `male`, `female`, or `unknown`. Blank values and unrecognized spellings were mapped to `unknown`. The script standardized 920 gender values.
3. **Age validation:** Six ages outside the range 10–100 were set to missing.
4. **Age imputation:** The 67 missing or invalid ages were filled using the median age of the corresponding city, calculated with `groupby` and `transform`.
5. **Signup date:** `signup_date` was converted to datetime before being saved in date format.

The cleaned customer table contains 1,200 records, with no missing ages and an age range of 18–60.

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

### Order Cleaning

The following rules were applied in the required order:

1. **Timestamp conversion:** `order_ts` was converted to datetime.
2. **Exact duplicates:** 119 exact duplicate rows were removed.
3. **Duplicate order IDs:** The latest `order_ts` record was retained for each order ID, removing 656 older records.
4. **Order amounts:** Unparseable amounts were converted to missing values. Missing amounts were calculated using `price_inr × quantity × (1 − discount_pct / 100)`, rounded to two decimal places. The cleaning script filled 112 missing amounts remaining after deduplication.
5. **Negative delivery times:** Negative values were set to missing. The cleaning script changed 105 such values in the deduplicated data.
6. **Completed-order delivery times:** Missing delivery times were filled using the median delivery time for the product category among completed orders, calculated using `groupby` and `transform`. The script filled 774 missing completed-order delivery times.
7. **Cancelled orders:** Delivery times for cancelled orders were left missing because these orders were not delivered.

The cleaned orders table contains 12,309 unique order IDs, with 11,110 completed orders and 1,199 cancelled orders. It contains no pending orders or missing order amounts. Completed orders have no missing delivery times, while all 1,199 cancelled orders have missing delivery times.

Completed-order revenue totals **INR 2,998,668.80**.

### Storage and Validation

The cleaned customer, product, and order tables were saved as CSV files in `data/clean/` and as SQLite tables named `customers`, `products`, and `orders` in `data/instamart.db`.

The `validate_data.py` script checks customer counts, age ranges, order counts, order statuses, missing-value rules, completed revenue, and database table row counts. Successful validation prints:

```text
ALL CHECKPOINTS PASSED
```

### SQL Analysis

Seven queries were saved in `sql/queries.sql` and executed from Python using `pandas.read_sql_query` against `data/instamart.db`.

| Query | Analysis | Reference result |
|---|---|---|
| Q1 | `WHERE` and `HAVING` | 30 customers |
| Q2 | Three-table join | 30 city/category rows; INR 2,998,668.80 total revenue |
| Q3 | Scalar subquery | 11 products |
| Q4 | `LEFT JOIN` | 25 customers without completed orders |
| Q5 | Common table expressions | 550 above-average spend customers; average INR 2,552.06 |
| Q6 | Ranking window functions | Correct top customer in each of six cities; 8 rows where `RANK()` differs from `ROW_NUMBER()` |
| Q7 | Daily running total | 183 days; final cumulative revenue INR 2,998,668.80 |

For Q6, ties can cause `RANK()` to assign the same rank to multiple customers, while `ROW_NUMBER()` assigns each row a unique sequential number.

### Pandas Cross-Check and Exploratory Analysis

A pandas pivot table using `pd.pivot_table(..., margins=True)` reproduced the city and category revenue results from Q2. Its grand total matched the SQL total of INR 2,998,668.80.

The completed-order city summary also matched the reference values: Bengaluru had revenue of INR 523,643.40, a mean order amount of INR 276.47, and 1,894 completed orders. Mumbai had revenue of INR 479,058.50, a mean of INR 297.00, and 1,613 completed orders.

The completed-order amount distribution has a mean of INR 269.91, a median of INR 135.00, and skewness of 4.61, indicating a strongly right-skewed distribution. Q1 is INR 64.80 and Q3 is INR 297.00. Using the 1.5×IQR rule, 897 orders exceed the upper fence of INR 645.30, while none fall below the lower fence of INR −283.50. These statistical outliers are not automatically errors.

The largest absolute correlation is between `order_amount` and `discount_pct` at −0.0951, followed by `order_amount` and `quantity` at +0.0873. Both relationships are weak; correlation of this size does not establish causation.

## Part B — Customer Segments and Lapse Prediction

### 1. Customer Segmentation Using RFM

The RFM analysis uses completed orders up to the snapshot date of `2025-10-01 00:00:00`. It calculates:

- **Recency:** Number of days since the customer's most recent completed order.
- **Frequency:** Number of distinct completed orders.
- **Monetary:** Total spending across completed orders.

The resulting RFM table contains 1,175 customers. The mean frequency is 9.46 orders, mean monetary value is ₹2,552.06, and median recency is 19 days.

The three RFM features were standardised using `StandardScaler`. K-means clustering was evaluated for K values from 2 to 8 using inertia and silhouette scores. K=4 was selected because it achieved the highest silhouette score of 0.5255 among the tested values while producing four actionable customer segments.

| Segment ID | Customers | Mean Recency (days) | Mean Frequency | Mean Monetary (₹) | Segment Name |
|---:|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Customers |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Loyal and Active |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Dormant Customers |

Cluster IDs are arbitrary labels and do not imply an ordering. The Premium flag is assigned to the segment with the highest mean monetary value.

PCA was applied to the standardised RFM matrix. The first component explains 63.43% of the variance, and the second explains 21.80%, giving a combined explained variance of 85.23%.

The segment assignments are saved in `outputs/customer_segments.csv`.

### 2. Lapse Prediction Problem

The lapse cohort contains customers with at least one completed order before `2025-09-01`. A customer is labelled as lapsed if they have no completed order from September 1 through September 30, 2025.

The cohort contains 1,156 customers: 443 lapsed customers (38.32%) and 713 non-lapsed customers.

Features are constructed using orders dated strictly before the September 1 cutoff. These include recency, frequency, monetary value, average discount percentage, cancellation rate, average delivery time, favourite product category, city, age, gender, and customer tenure.

The model uses a stratified 80/20 train/test split with `random_state=42`, resulting in 924 training rows and 232 test rows.

### 3. Leakage-Safe Preprocessing

Numeric features are imputed using the median and standardised using `StandardScaler`. City, gender, and favourite category are imputed using the most frequent value and one-hot encoded using `OneHotEncoder(drop="first")`.

Preprocessing is implemented using scikit-learn pipelines. Every encoder and scaler is fitted on training rows only and then applied to the test rows. During grid search, preprocessing is refitted within each cross-validation training fold.

The fitted scaler's training-only mean for frequency is 8.1147, compared with the full-cohort mean of 8.0285. These values differ because the scaler is fitted on the training subset rather than the full cohort.

No feature column contained missing values after preprocessing. No orders dated on or after `2025-09-01` entered the model features, and the customer segment labels were not used as predictive features.

### 4. Model Tuning and Evaluation

A baseline that always predicts non-lapsed achieves 61.64% accuracy but an F1 score of zero for the lapsed class. This demonstrates why accuracy alone is insufficient for a retention use case.

Both models were tuned using five-fold `GridSearchCV` with F1 scoring on the training set only.

| Model | Best Parameters | Best CV F1 | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | C = 100 | 0.6506 | 0.6444 | 0.6517 | 0.6480 |
| Random Forest | max_depth = 8, min_samples_leaf = 10 | 0.6811 | 0.6591 | 0.6517 | 0.6554 |

The random forest was selected because it achieved the higher test F1 score. Its confusion matrix, in the order `[[TN, FP], [FN, TP]]`, is `[[113, 30], [31, 58]]`.

Missing a lapsing customer may result in lost future revenue, whereas targeting a loyal customer can waste campaign resources or incentives. The selected random forest offers a balance between identifying lapsing customers and limiting unnecessary retention targeting.

### 5. Class-Balance Experiment

The logistic regression training data was oversampled by randomly duplicating lapsed training rows with replacement until both classes contained 570 rows. The original test set was left unchanged.

| Metric | Original Logistic Regression | Balanced Logistic Regression |
|---|---:|---:|
| Precision | 0.6444 | 0.5891 |
| Recall | 0.6517 | 0.8539 |
| F1 | 0.6480 | 0.6972 |

Oversampling substantially increased recall but reduced precision. This means the retention team can identify a larger proportion of lapsing customers, although more loyal customers may also receive retention offers.

### 6. Model Interpretation and Saved Outputs

The five most important random forest features were frequency, monetary value, recency, average delivery time, and cancellation rate.

The logistic regression coefficients for recency and monetary value were positive, while frequency had a negative coefficient. These signs describe associations with predicted lapse odds while holding other model features constant; they do not establish causation.

The selected random forest is saved to `outputs/lapse_model.joblib`. The saved model was reloaded and produced exactly the same test predictions as the original model.

The ten highest predicted lapse probabilities among Premium customers in the test set are saved to `outputs/retention_list.csv`, with the columns `customer_id`, `lapse_probability`, and `actually_lapsed`. One of the ten listed customers actually lapsed.

**Important limitation:** The Premium flag is derived from whole-window RFM, which includes September activity used to define the lapse label. Therefore, the retention list is illustrative and is not a leakage-free evaluation.


## Part C — Support Ticket Triage with Guardrails

### 1. Input Guardrails and Privacy

Processed all 164 support tickets in their original file order. Four empty or whitespace-only tickets were skipped, and four tickets exceeding the 400-token input limit were flagged without calling the model.

Personal information was redacted before constructing model prompts using the required regular expressions, in this order:

- Email addresses: 55 substitutions
- Phone numbers: 91 substitutions
- Order IDs: 132 substitutions
- **Total substitutions: 278**

Assertions verified that no email, phone number, or order ID matching the specified patterns remained in the redacted text. The model's user message contained only the redacted ticket text.

### 2. Prompts and Retry Handling

Implemented zero-shot and few-shot system prompts. Both require classification into billing, delivery, technical, or fraud and a JSON-only response.

Transient failures are handled by retry logic with up to three retries, exponential backoff, and jitter. The mock was reset before each independent run. All 19 simulated transient failures were recovered, and retry logs recorded the ticket ID, retry number, and wait duration.

### 3. Output Validation

Responses were parsed as JSON first. Valid categories received the `classified_json` status. Non-JSON responses were checked using lowercase word extraction and received `classified_keyword` only when exactly one allowed category was present. Invalid JSON categories and ambiguous or unrecognised responses were flagged for human review.

| Status | Tickets |
|---|---:|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |
| **Total** | **164** |

### 4. Zero-Shot and Few-Shot Comparison

Both runs classified 147 tickets.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.7% | 100.0% |
| Total tokens | 12,052 | 22,192 |
| Tokens per classified ticket | 81.99 | 150.97 |

The few-shot run used 10,140 additional tokens and improved accuracy by 14.3 percentage points. Zero-shot produced 21 classification errors: 11 fraud tickets and 10 technical tickets were incorrectly labelled as billing. In this experiment, the extra few-shot cost was worthwhile when classification accuracy and avoiding these errors were more important than token minimisation.

**Zero-shot confusion matrix** (rows = actual categories; columns = predicted categories):

| Actual \ Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

### 5. Token Budget and Routing

The zero-shot budget demonstration used a 6,000-token budget and saved `outputs/triage_budget_demo.csv`. It stopped making further calls after the running token total reached 6,046, leaving 78 tickets with `budget_exhausted`.

The final few-shot run used a 100,000-token budget and saved `outputs/triage_results.csv`. The output contains 164 rows and these columns: `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`.

Routing follows the required rules:

- Empty, long, invalid-output, and budget-exhausted tickets → `human_review`, NORMAL.
- Validly classified fraud tickets → `security_team`, HIGH.
- Other valid classifications for premium customers → `premium_desk`, HIGH.
- Remaining valid classifications → `support_queue`, NORMAL.

The final routing counts were:

| Route | Tickets |
|---|---:|
| `support_queue` | 102 |
| `security_team` | 34 |
| `human_review` | 17 |
| `premium_desk` | 11 |
| **Total** | **164** |

### 6. Limitations and Human Review

Only responses passing the JSON or unambiguous-keyword validation rules were treated as classified. Invalid outputs and tickets blocked by input or token-budget guardrails were routed to human review rather than assigned a guessed category. The category validation checks the response format and allowed labels; it does not independently verify that the model's reasoning is factually correct. The supplied mock LLM is deterministic and offline, so its measured performance should not be assumed to represent a hosted model or real-world support traffic.

## Part D — Policy RAG and ReAct Support Agent

### 1. Policy Retrieval-Augmented Generation (RAG)

The policy RAG pipeline retrieves relevant information from five policy documents stored in `data/policies/`. The documents cover delivery, escalation, payment, privacy, and refund policies.

The implementation in `policy_rag.py` uses the following configuration:

- **Chunk size:** 40 words
- **Chunk overlap:** 10 words
- **Retrieval method:** TF-IDF with cosine similarity
- **Retrieved context:** Top three matching chunks
- **Language model:** Supplied offline mock LLM

The five policy documents produce 15 chunks in total. Chunk overlap helps preserve context between adjacent sections. For each question, the retriever ranks policy chunks by similarity and supplies the top three results to the answer-generation step.

The system prompt instructs the assistant to answer using the supplied policy context. When the information cannot be found, the expected fallback is:

`I could not find that information in the policy documents.`

### 2. RAG Evaluation

The RAG implementation was evaluated using the 14 questions in `data/raw/policy_eval.csv`.

| Evaluation measure | Overlap enabled | Overlap disabled |
|---|---:|---:|
| Expected document matches | 12/12 | 12/12 |
| Expected fact matches | 12/12 | 10/12 |
| Expected fallback matches | 2/2 | 2/2 |

With overlap enabled, all 12 questions with expected policy information matched the expected facts. Without overlap, two expected facts were not matched: the delivery-time condition for a full refund and the refund timing for a failed payment that was deducted.

These results demonstrate the value of chunk overlap for this evaluation set. They do not establish general performance on unseen policies or questions.

### 3. Agent Tools and Refund Guardrails

`agent_tools.py` exposes five tools to the agent:

| Tool | Purpose |
|---|---|
| `get_order` | Retrieves an order from the SQLite database and reports when it cannot be found. |
| `search_policy` | Retrieves relevant policy chunks for a query. |
| `calculate_refund` | Checks refund eligibility and calculates the permitted amount. |
| `issue_refund` | Validates the refund calculation before recording a refund decision. |
| `escalate_to_human` | Records a decision requiring human review. |

The refund workflow checks order details and eligibility before authorizing a refund. Orders requiring approval, including refunds above ₹1,000, are escalated rather than automatically refunded. The implementation also checks the refund amount against the calculated amount.

The database does not contain delivery dates needed to verify the seven-day refund window. Consequently, the agent requires explicit request and delivery dates for that check when processing a real database order. The test harness supplies controlled dates for its refund scenarios.

Refund and escalation decisions are recorded in `outputs/refund_ledger.jsonl`.

### 4. ReAct Agent and Execution Limits

`react_agent.py` implements a ReAct-style loop in which the model selects a tool, receives its observation, and decides what to do next.

The agent includes these execution limits:

- Maximum of six iterations by default.
- Default token limit of 6,000.
- Default execution time limit of 10 seconds.
- Checkpoint persistence after tool observations.
- Resume support through `resume_agent(run_id)`.
- A final refund status is accepted only when the issue-refund tool confirms a positive refund.

Run history is recorded in `outputs/agent_runs.jsonl`. Checkpoint files are stored in the top-level `state/` directory.

### 5. Scenario Tests

The seven supplied scenarios in `data/raw/agent_scenarios.csv` cover successful refunds, escalation, denial, missing orders, and iteration-limit fallback.

| Scenario | Expected outcome | Tested outcome | Result |
|---|---|---|---|
| S1 | REFUND | REFUND | PASS |
| S2 | REFUND | REFUND | PASS |
| S3 | ESCALATE | ESCALATE | PASS |
| S4 | DENIED | DENIED | PASS |
| S5 | DENIED | DENIED | PASS |
| S6 | NOT_FOUND | NOT_FOUND | PASS |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | PASS |

The tests also verified token-limit fallback and timeout fallback. A simulated interruption after the second saved tool observation was resumed successfully, and the agent completed the remaining workflow. The test harness reported **10/10 checks passed**.

The successful refund scenarios were tested with explicit fixture dates. The test results therefore validate the configured scenarios and guardrails, not live payment processing.

### 6. ReAct Versus Plan-and-Execute

ReAct combines reasoning, tool selection, and observation-driven decisions in an iterative loop. After each tool returns an observation, the agent can use that result to choose the next action. This is useful for support workflows where order status, policy eligibility, and refund approval determine what should happen next.

A plan-and-execute approach separates planning from execution. The system first creates a sequence of steps and then carries them out, potentially adjusting the plan if a step fails. This can make the workflow easier to inspect when the required steps are known in advance.

For this project, ReAct supports conditional decisions such as stopping when an order is missing, denying an ineligible refund, or escalating a high-value refund. However, iterative tool use can increase latency and token consumption. The iteration, token, and time limits help constrain execution. Checkpointing also supports recovery after an interruption.

The implementation demonstrates a bounded, tool-using agent with deterministic offline tests; it does not establish that ReAct is universally better than plan-and-execute.

## Limitations and Next Steps

- **Offline model:** The supplied mock LLM is deterministic. Results may differ substantially with a hosted language model or real support traffic.
- **Limited evaluation data:** The policy RAG evaluation contains 14 questions, and the agent scenarios cover seven predefined cases. Broader tests are needed before production use.
- **Policy retrieval:** TF-IDF similarity may miss relevant information when wording differs significantly from the policy documents.
- **Order data:** Missing delivery dates prevent independent verification of the seven-day refund window using database records alone.
- **Refund processing:** The project records refund decisions in a local JSONL ledger. It does not connect to a real payment gateway or issue live refunds.
- **Human approval:** High-value refunds and uncertain outcomes should remain subject to human review.
- **Potential improvements:** Add more evaluation cases, improve retrieval quality, test against a real language model, integrate authoritative delivery timestamps, and introduce authenticated payment-system integration with audit controls.