# quick_overlap_check.py
# Confirm the trimmed files cover our labeled population as expected.

import duckdb

con = duckdb.connect()

total_labeled = con.execute("""
    SELECT COUNT(DISTINCT msno) FROM read_csv_auto('train_v2.csv')
""").fetchone()[0]

log_users = 781835    # from trim output
txn_users = 936986    # from trim output

# Users in transactions but NOT in logs — these will have null engagement features
txn_not_in_logs = con.execute("""
    SELECT COUNT(DISTINCT t.msno)
    FROM read_csv_auto('transactions_trimmed.csv') t
    LEFT JOIN read_csv_auto('user_logs_trimmed.csv') l
      ON t.msno = l.msno
    WHERE l.msno IS NULL
""").fetchone()[0]

# Users in train_v2 with neither logs nor transactions
neither = con.execute("""
    SELECT COUNT(DISTINCT v.msno)
    FROM read_csv_auto('train_v2.csv') v
    LEFT JOIN read_csv_auto('transactions_trimmed.csv') t ON v.msno = t.msno
    LEFT JOIN read_csv_auto('user_logs_trimmed.csv')    l ON v.msno = l.msno
    WHERE t.msno IS NULL AND l.msno IS NULL
""").fetchone()[0]

print(f"Total labeled users (train_v2)          : {total_labeled:>8,}")
print(f"Labeled users with log activity          : {log_users:>8,}  ({log_users/total_labeled*100:.1f}%)")
print(f"Labeled users with transaction activity  : {txn_users:>8,}  ({txn_users/total_labeled*100:.1f}%)")
print(f"In transactions but no logs              : {txn_not_in_logs:>8,}  ({txn_not_in_logs/total_labeled*100:.1f}%)")
print(f"In neither file (label only)             : {neither:>8,}  ({neither/total_labeled*100:.1f}%)")
print(f"\nNote: 'label only' users will carry null engagement and")
print(f"subscription features — handled via imputation in Step 4.")

con.close()

"""
96.5% of labeled users have transaction activity- nearly everyone has a subscription record in the trimmed window. That's expected; 
These users were selected into train_v2 precisely because their membership expired in February, so they almost certainly have a February transaction.

80.5% have log activity- the remaining 19.5% subscribed but didn't listen in Jan to Feb period. 
These are passive subscribers who pay but don't engage and are a distinct churn profile. 
They won't have null features because their behavior was genuinely zero. 

The 19.5% is broken down to the following.

18.9% are in transactions but not logs-these are the passive subscribers above. 
They'll get engagement features imputed to zero (not median), because zero is the true value.

0.6% are in neither file (label only)- 6,014 users.
These are users whose membership expired in February but had no recorded activity in our 60-day window at all. 
They'll carry nulls across all behavioral features. At 0.6% they won't materially affect the model 
"""