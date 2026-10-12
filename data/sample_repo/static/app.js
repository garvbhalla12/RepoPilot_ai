// Browser client for the ShelfLife API.
const API = "/api";
let sessionToken = null;

async function login(email, password) {
  const res = await fetch(`${API}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error("Login failed");
  sessionToken = (await res.json()).token;
}

async function searchBooks(query) {
  const res = await fetch(`${API}/books?q=${encodeURIComponent(query)}`);
  return res.json();
}

async function borrowBook(bookId) {
  const res = await fetch(`${API}/loans`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${sessionToken}` },
    body: JSON.stringify({ book_id: bookId }),
  });
  return res.json();
}
