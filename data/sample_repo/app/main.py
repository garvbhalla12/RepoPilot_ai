"""HTTP API routes for ShelfLife (Flask)."""
from datetime import date

from flask import Flask, jsonify, request

from app import auth, loans
from app.database import get_connection, init_db

app = Flask(__name__)


def current_member():
    """Return the token payload from the Authorization header, or None if missing/invalid."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return auth.verify_token(header[len("Bearer "):])


@app.route("/api/register", methods=["POST"])
def register():
    """Create a member account from an email and password."""
    data = request.get_json(force=True)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO members (email, password_hash) VALUES (?, ?)",
            (data["email"], auth.hash_password(data["password"])),
        )
    return jsonify({"status": "created"}), 201


@app.route("/api/login", methods=["POST"])
def login():
    """Verify credentials and return a signed session token."""
    data = request.get_json(force=True)
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM members WHERE email = ?", (data["email"],)).fetchone()
    if row is None or not auth.verify_password(data["password"], row["password_hash"]):
        return jsonify({"error": "invalid credentials"}), 401
    return jsonify({"token": auth.issue_token(row["id"], bool(row["is_admin"]))})


@app.get("/api/books")
def list_books():
    """List books, optionally filtered by a title search term (?q=)."""
    term = f"%{request.args.get('q', '')}%"
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM books WHERE title LIKE ?", (term,)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/loans")
def create_loan():
    """Borrow a book. Requires a valid bearer token."""
    member = current_member()
    if member is None:
        return jsonify({"error": "authentication required"}), 401
    try:
        loan_id = loans.borrow_book(member["sub"], request.get_json(force=True)["book_id"], date.today())
    except loans.LoanError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"loan_id": loan_id}), 201


@app.post("/api/loans/<int:loan_id>/return")
def return_loan(loan_id: int):
    """Return a borrowed book and report any late fee."""
    if current_member() is None:
        return jsonify({"error": "authentication required"}), 401
    try:
        fee = loans.return_book(loan_id, date.today())
    except loans.LoanError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"late_fee": fee})


if __name__ == "__main__":
    init_db()
    app.run(port=5000)
