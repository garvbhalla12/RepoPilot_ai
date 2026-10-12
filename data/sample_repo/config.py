"""Application configuration loaded from environment variables."""
import os

DB_PATH = os.getenv("SHELFLIFE_DB_PATH", "shelflife.db")
SECRET_KEY = os.getenv("SHELFLIFE_SECRET", "")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))
LOAN_PERIOD_DAYS = 14
LATE_FEE_PER_DAY = 0.25
MAX_ACTIVE_LOANS = 5
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "25"))
