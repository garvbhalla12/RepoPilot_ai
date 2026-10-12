"""Tests for the late-fee rule."""
from datetime import date

from app.loans import calculate_late_fee


def test_no_fee_when_returned_on_time():
    assert calculate_late_fee(date(2025, 1, 10), date(2025, 1, 10)) == 0


def test_fee_accumulates_per_day():
    assert calculate_late_fee(date(2025, 1, 10), date(2025, 1, 14)) == 1.0
