"""Borrowing and returning rules, including late-fee calculation."""
from datetime import date, timedelta

import config
from app.database import get_connection


class LoanError(Exception):
    """Raised when a borrow or return request violates a business rule."""


def calculate_late_fee(due_on: date, returned_on: date) -> float:
    """Late fee is config.LATE_FEE_PER_DAY for every day returned after the due date."""
    days_late = (returned_on - due_on).days
    return round(max(days_late, 0) * config.LATE_FEE_PER_DAY, 2)


def borrow_book(member_id: int, book_id: int, today: date) -> int:
    """Create a loan if the book has copies left and the member is under MAX_ACTIVE_LOANS."""
    with get_connection() as conn:
        active = conn.execute(
            "SELECT COUNT(*) FROM loans WHERE member_id = ? AND returned_on IS NULL", (member_id,)
        ).fetchone()[0]
        if active >= config.MAX_ACTIVE_LOANS:
            raise LoanError("Member has reached the maximum number of active loans")
        book = conn.execute("SELECT copies_available FROM books WHERE id = ?", (book_id,)).fetchone()
        if book is None or book["copies_available"] < 1:
            raise LoanError("No copies available")
        due = today + timedelta(days=config.LOAN_PERIOD_DAYS)
        cur = conn.execute(
            "INSERT INTO loans (book_id, member_id, borrowed_on, due_on) VALUES (?, ?, ?, ?)",
            (book_id, member_id, today.isoformat(), due.isoformat()),
        )
        conn.execute("UPDATE books SET copies_available = copies_available - 1 WHERE id = ?", (book_id,))
        return cur.lastrowid


def return_book(loan_id: int, today: date) -> float:
    """Mark a loan returned, restock the book and return the late fee owed."""
    with get_connection() as conn:
        loan = conn.execute("SELECT * FROM loans WHERE id = ?", (loan_id,)).fetchone()
        if loan is None or loan["returned_on"] is not None:
            raise LoanError("Loan not found or already returned")
        fee = calculate_late_fee(date.fromisoformat(loan["due_on"]), today)
        conn.execute("UPDATE loans SET returned_on = ? WHERE id = ?", (today.isoformat(), loan_id))
        conn.execute("UPDATE books SET copies_available = copies_available + 1 WHERE id = ?", (loan["book_id"],))
        return fee
