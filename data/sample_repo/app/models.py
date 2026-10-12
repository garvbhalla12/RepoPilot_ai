"""Plain dataclasses describing the domain objects."""
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Book:
    id: int
    title: str
    author: str
    isbn: str
    copies_available: int


@dataclass
class Member:
    id: int
    email: str
    password_hash: str
    is_admin: bool = False


@dataclass
class Loan:
    id: int
    book_id: int
    member_id: int
    borrowed_on: date
    due_on: date
    returned_on: Optional[date] = None

    def is_overdue(self, today: date) -> bool:
        """A loan is overdue when it has not been returned and the due date has passed."""
        return self.returned_on is None and today > self.due_on
