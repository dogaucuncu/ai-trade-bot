import sqlite3
conn = sqlite3.connect('data/tradebot.db')
c = conn.cursor()
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
for t in tables:
    name = t[0]
    count = c.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"Table {name}: {count} rows")
