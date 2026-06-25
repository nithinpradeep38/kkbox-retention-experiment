# Compare how many train_v2 users we capture at different lookback windows.
# This helps us pick the right cutoff — not too short, not unnecessarily long.

import duckdb

con = duckdb.connect()

print("train_v2 users captured at different lookback windows:")
print("(total labeled users = 970,960)\n")

windows = [
    ("Last 30 days  (Feb 1  – Feb 28)", 20170201, 20170228),
    ("Last 60 days  (Jan 1  – Feb 28)", 20170101, 20170228),
    ("Last 90 days  (Dec 1  – Feb 28)", 20161201, 20170228),
    ("Last 180 days (Sep 1  – Feb 28)", 20160901, 20170228),
    ("Last 365 days (Mar 1  – Feb 28)", 20160301, 20170228),
]

for label, start, end in windows:
    result = con.execute(f"""
        SELECT COUNT(DISTINCT l.msno) AS users_captured
        FROM read_csv_auto('user_logs.csv')  l
        JOIN read_csv_auto('train_v2.csv')   t
          ON l.msno = t.msno
        WHERE l.date >= {start}
          AND l.date <= {end}
    """).fetchone()[0]

    pct = result / 970960 * 100
    print(f"  {label} : {result:>7,} users  ({pct:.1f}%)")

con.close()
"""
The returns diminish sharply after 60 days. 
Going from 30→60 days buys you 26k users. 
Going from 60→90 buys 12k. 
Going from 90→365 days — an extra 9 months of data — only buys another 38k users while multiplying the data volume dramatically.
The ~18% of users with no logs in any window are likely passive subscribers who don't use the app. They'll appear with null engagement features, which the model will handle.

Jan 1 – Feb 28 is the right cutoff for this project. Here's my reasoning.
- Captures 80.5% of labeled users, essentially the same coverage as 90 days
- Cuts log volume roughly in half versus 90 days (60 days of 2017's rate ≈ ~34M rows vs ~51M)
- Recent behavior is more predictive of near-term churn than behavior from 3+ months ago
- The marginal 12k users gained by going to 90 days don't justify the extra volume
"""