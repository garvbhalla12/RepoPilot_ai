"""Overdue reminder emails sent through an SMTP server."""
import smtplib
from datetime import date
from email.message import EmailMessage

import config
from app.database import get_connection


def build_reminder(email: str, title: str, due_on: str) -> EmailMessage:
    """Compose the reminder message for one overdue book."""
    msg = EmailMessage()
    msg["To"] = email
    msg["From"] = "library@shelflife.example"
    msg["Subject"] = f"Overdue: {title}"
    msg.set_content(f"'{title}' was due on {due_on}. Please return it to avoid late fees.")
    return msg


def send_overdue_reminders(today: date) -> int:
    """Email every member with an overdue loan. Returns how many reminders were sent."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT m.email, b.title, l.due_on FROM loans l "
            "JOIN members m ON m.id = l.member_id JOIN books b ON b.id = l.book_id "
            "WHERE l.returned_on IS NULL AND l.due_on < ?", (today.isoformat(),),
        ).fetchall()
    sent = 0
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as smtp:
        for row in rows:
            smtp.send_message(build_reminder(row["email"], row["title"], row["due_on"]))
            sent += 1
    return sent
