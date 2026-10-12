# ShelfLife — sample library-lending API

ShelfLife is a tiny, original demo service used by RepoPilot AI for its evaluation set.
It lets members borrow books, tracks due dates and sends overdue reminders.

## Layout
- `app/main.py` — HTTP routes (Flask)
- `app/auth.py` — password hashing and signed session tokens
- `app/database.py` — SQLite connection and schema
- `app/models.py` — dataclasses for books, members and loans
- `app/loans.py` — borrowing and returning rules, late-fee calculation
- `app/notifier.py` — overdue reminder emails
- `app/legacy.py` — old import helper kept for reference (contains intentionally risky code for scanner demos)
- `static/app.js` — browser client

## Configuration
Settings are read from environment variables in `config.py` (`SHELFLIFE_DB_PATH`, `SHELFLIFE_SECRET`, `SMTP_HOST`).
