# Before trimming, understand the date distribution in both files.
# We want to make sure our cutoff captures meaningful data.

import duckdb

con = duckdb.connect()

print("=" * 55)
print("USER LOGS — date distribution")
print("=" * 55)

# Overall date range
con.execute("""
    SELECT
        MIN(date) AS earliest_date,
        MAX(date) AS latest_date,
        COUNT(*)  AS total_rows
    FROM read_csv_auto('user_logs.csv')
""").df().pipe(print)

# Row count by year — how much data lives in each year?
print("\nRows by year:")
con.execute("""
    SELECT
        CAST(date / 10000 AS INT) AS year,
        COUNT(*)                  AS row_count,
        COUNT(DISTINCT msno)      AS distinct_users
    FROM read_csv_auto('user_logs.csv')
    GROUP BY year
    ORDER BY year
""").df().pipe(print)

print("\n" + "=" * 55)
print("TRANSACTIONS — date distribution")
print("=" * 55)

con.execute("""
    SELECT
        MIN(transaction_date) AS earliest_txn,
        MAX(transaction_date) AS latest_txn,
        COUNT(*)              AS total_rows
    FROM read_csv_auto('transactions.csv')
""").df().pipe(print)

print("\nRows by year:")
con.execute("""
    SELECT
        CAST(transaction_date / 10000 AS INT) AS year,
        COUNT(*)                              AS row_count,
        COUNT(DISTINCT msno)                  AS distinct_users
    FROM read_csv_auto('transactions.csv')
    GROUP BY year
    ORDER BY year
""").df().pipe(print)

print("\n" + "=" * 55)
print("USER OVERLAP — logs vs train_v2")
print("=" * 55)

# How many train_v2 users appear in logs at all?
con.execute("""
    SELECT
        COUNT(DISTINCT l.msno) AS log_users_in_train
    FROM read_csv_auto('user_logs.csv')   l
    JOIN read_csv_auto('train_v2.csv')    t
      ON l.msno = t.msno
""").df().pipe(print)

# How many train_v2 users appear in the last 90 days of logs?
print("\ntrain_v2 users with activity in last 90 days (Dec 1 – Feb 28):")
con.execute("""
    SELECT
        COUNT(DISTINCT l.msno) AS users_with_recent_logs
    FROM read_csv_auto('user_logs.csv')  l
    JOIN read_csv_auto('train_v2.csv')   t
      ON l.msno = t.msno
    WHERE l.date >= 20161201
      AND l.date <= 20170228
""").df().pipe(print)

con.close()


"""
What we're looking at
User logs — 392M rows total

2015: 161M rows, 2.8M users
2016: 197M rows, 3.1M users
2017: 34M rows, 1.3M users (Jan–Feb only)

The overlap numbers are the key finding:

train_v2 has 970,960 users (we know this from earlier)
Of those, 850,296 appear anywhere in user_logs.csv
Of those, 793,729 have activity specifically in the last 90 days (Dec 1 – Feb 28)

So 793,729 / 970,960 = 81.7% of our labeled users have recent log activity. The remaining ~18% either never logged listening activity or dropped off before December. That's fine — we'll carry them through with null engagement features.

"""