# 00_local_trim.py
#
# PURPOSE
# -------
# Reduce user_logs.csv (28 GB) and transactions.csv (1.6 GB) to uploadable
# size before ingestion into Databricks.

# USER FILTER
# -----------
# We keep only msno values present in train_v2.csv. Logs for users with no
# churn label are not useful for model training.
#
# SAMPLING RULE
# -------------
# We filter by msno FIRST (join to train_v2), then by date.
# Never sample log rows independently — that would bias per-user aggregates
# by randomly dropping days from a user's history.
#
# OUTPUT
# ------
# user_logs_trimmed.csv      → upload to Databricks landing volume
# transactions_trimmed.csv   → upload to Databricks landing volume

import duckdb
import time

LOGS_CUTOFF_START   = 20170101   # Jan 1, 2017
LOGS_CUTOFF_END     = 20170228   # Feb 28, 2017 (end of pre-expiry window)
TXN_CUTOFF_START    = 20170101   # same window for consistency
TXN_CUTOFF_END      = 20170228

con = duckdb.connect()

#USER LOGS

print("=" * 60)
print("Trimming user_logs.csv")
print(f"  Keeping: Jan 1 2017 – Feb 28 2017")
print(f"  Keeping: only msno values in train_v2.csv")
print("=" * 60)

t0 = time.time()

con.execute(f"""
    COPY (
        SELECT l.*
        FROM read_csv_auto('user_logs.csv')  l
        JOIN read_csv_auto('train_v2.csv')   t
          ON l.msno = t.msno
        WHERE l.date >= {LOGS_CUTOFF_START}
          AND l.date <= {LOGS_CUTOFF_END}
    )
    TO 'user_logs_trimmed.csv' (HEADER, DELIMITER ',')
""")

elapsed = time.time() - t0
print(f"  Done in {elapsed/60:.1f} minutes")

# Quick sanity check on output
result = con.execute("""
    SELECT
        COUNT(*)              AS total_rows,
        COUNT(DISTINCT msno)  AS distinct_users,
        MIN(date)             AS earliest_date,
        MAX(date)             AS latest_date
    FROM read_csv_auto('user_logs_trimmed.csv')
""").df()

print("\n  Sanity check — user_logs_trimmed.csv:")
print(result.to_string(index=False))

#TRANSACTIONS

print("\n" + "=" * 60)
print("Trimming transactions.csv")
print(f"  Keeping: Jan 1 2017 – Feb 28 2017")
print(f"  Keeping: only msno values in train_v2.csv")
print("=" * 60)

t0 = time.time()

con.execute(f"""
    COPY (
        SELECT t.*
        FROM read_csv_auto('transactions.csv')  t
        JOIN read_csv_auto('train_v2.csv')       v
          ON t.msno = v.msno
        WHERE t.transaction_date >= {TXN_CUTOFF_START}
          AND t.transaction_date <= {TXN_CUTOFF_END}
    )
    TO 'transactions_trimmed.csv' (HEADER, DELIMITER ',')
""")

elapsed = time.time() - t0
print(f"  Done in {elapsed/60:.1f} minutes")

result = con.execute("""
    SELECT
        COUNT(*)              AS total_rows,
        COUNT(DISTINCT msno)  AS distinct_users,
        MIN(transaction_date) AS earliest_txn,
        MAX(transaction_date) AS latest_txn
    FROM read_csv_auto('transactions_trimmed.csv')
""").df()

print("\n  Sanity check — transactions_trimmed.csv:")
print(result.to_string(index=False))

# FINAL SUMMARY

print("\n" + "=" * 60)
print("TRIM COMPLETE")
print("=" * 60)
print("Next step: upload both trimmed files to Databricks:")
print("  /Volumes/churn_project/bronze/landing/user_logs_trimmed.csv")
print("  /Volumes/churn_project/bronze/landing/transactions_trimmed.csv")

con.close()

"""
user_logs_trimmed.csv

- 24.9M rows, 781,835 users, Jan 1 to Feb 28 ✓
- 781,835 matches exactly what our window exploration predicted for 60 days ✓
- Down from 392M rows to 25M: 94% reduction in volume

transactions_trimmed.csv

- 1.77M rows, 936,986 users, Jan 1 to Feb 28 ✓
- Transactions has more users (936k) than logs (781k) — makes sense, some users have active subscriptions but aren't listening
- Down from 21.5M rows to 1.77M: 92% reduction in volume

"""

# DATA QUALITY NOTES (from pre-upload exploration)

# train_v2 total labeled users      : 970,960
# Users with log activity (60d)     : 781,835  (80.5%)
# Users with transaction activity   : 936,986  (96.5%)
# Passive subscribers (txn, no log) : 183,111  (18.9%)- zero engagement, not missing data.
# Label-only users (no activity)    :   6,014   (0.6%) — will carry null features
#
# NOTE: passive subscribers (no log activity) should receive
# zero-imputed engagement features, NOT median imputation.
# Their zero activity is the signal.
