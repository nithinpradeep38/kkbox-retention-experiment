# Quick local check (run this on your laptop with DuckDB, or upload train.csv
# to Databricks temporarily to check) — let's verify train.csv's cohort timing

import duckdb
con = duckdb.connect()

# How many train.csv users have a transaction expiring in January 2017?
result = con.execute("""
    SELECT COUNT(DISTINCT t.msno) as users_with_jan_expiry
    FROM read_csv_auto('transactions.csv') t
    JOIN read_csv_auto('train.csv') tr ON t.msno = tr.msno
    WHERE t.membership_expire_date >= 20170101
      AND t.membership_expire_date <= 20170131
""").fetchone()[0]

total_train_v1 = con.execute("""
    SELECT COUNT(DISTINCT msno) FROM read_csv_auto('train.csv')
""").fetchone()[0]

print(f"train.csv total users: {total_train_v1:,}")
print(f"train.csv users with a transaction expiring in Jan 2017: {result:,}")
print(f"Percentage: {result/total_train_v1*100:.1f}%")