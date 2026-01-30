from flask import Flask, request, jsonify, g

import sqlite3

from typing import Dict, Any
 
 
USE_CORS = False
 
DATABASE = "loan_applicants.db"
 
app = Flask(__name__)
 
 
# ---------- DB Utilities ----------
 
def get_db():

    """

    Get a per-request DB connection stored in Flask's 'g'.

    Sets row_factory to sqlite3.Row so rows behave like dicts.

    """

    if "db" not in g:

        conn = sqlite3.connect(DATABASE)

        conn.row_factory = sqlite3.Row

        g.db = conn

    return g.db
 
@app.teardown_appcontext

def close_db(exception=None):

    db = g.pop("db", None)

    if db is not None:

        db.close()
 
def init_db():

    with sqlite3.connect(DATABASE) as conn:

        c = conn.cursor()

        c.execute(

            """

            CREATE TABLE IF NOT EXISTS applicants (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                Married TEXT,

                Education TEXT,

                Self_Employed TEXT,

                ApplicantIncome INTEGER,

                CoapplicantIncome INTEGER,

                LoanAmount INTEGER,

                Loan_Amount_Term INTEGER,

                Credit_History INTEGER,

                Property_Area TEXT,

                Prediction INTEGER

            )

            """

        )

        # Optional: useful indexes for filtering/sorting (adjust as needed)

        c.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON applicants(Prediction)")

        c.execute("CREATE INDEX IF NOT EXISTS idx_property_area ON applicants(Property_Area)")

        conn.commit()
 
# Initialize DB on import

init_db()
 
 
# ---------- Helpers ----------
 
REQUIRED_FIELDS = [

    "Married",

    "Education",

    "Self_Employed",

    "ApplicantIncome",

    "CoapplicantIncome",

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History",

    "Property_Area",

    "Prediction",

]
 
# Whitelist for safe updates (avoid SQL injection on column names)

UPDATABLE_COLUMNS = {

    "Married",

    "Education",

    "Self_Employed",

    "ApplicantIncome",

    "CoapplicantIncome",

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History",

    "Property_Area",

    "Prediction",

}
 
def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:

    return {k: row[k] for k in row.keys()}
 
def validate_payload(payload: Dict[str, Any], required: bool = True):

    if not isinstance(payload, dict):

        return False, "Invalid JSON body."
 
    missing = [f for f in REQUIRED_FIELDS if required and f not in payload]

    if missing:

        return False, f"Missing required fields: {', '.join(missing)}"
 
    # Basic type checks (minimal; extend as needed)

    ints = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Prediction"]

    for f in ints:

        if f in payload and payload[f] is not None and not isinstance(payload[f], int):

            return False, f"Field '{f}' must be an integer."
 
    # Strings (allow None or str)

    strs = ["Married", "Education", "Self_Employed", "Property_Area"]

    for f in strs:

        if f in payload and payload[f] is not None and not isinstance(payload[f], str):

            return False, f"Field '{f}' must be a string."
 
    return True, None
 
 
# ---------- Routes ----------
 
@app.route("/health", methods=["GET"])

def health():

    return jsonify({"status": "ok"}), 200
 
 
@app.route("/applicants", methods=["GET"])

def get_applicants():

    """

    Optional query params: limit, offset, prediction, property_area

    e.g., /applicants?limit=50&offset=0&prediction=1&property_area=Urban

    """

    db = get_db()

    limit = request.args.get("limit", type=int) or 100

    offset = request.args.get("offset", type=int) or 0

    prediction = request.args.get("prediction", type=int)

    property_area = request.args.get("property_area", type=str)
 
    sql = "SELECT * FROM applicants"

    clauses = []

    params = []
 
    if prediction is not None:

        clauses.append("Prediction = ?")

        params.append(prediction)

    if property_area:

        clauses.append("Property_Area = ?")

        params.append(property_area)
 
    if clauses:

        sql += " WHERE " + " AND ".join(clauses)
 
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"

    params.extend([limit, offset])
 
    rows = db.execute(sql, params).fetchall()

    return jsonify([row_to_dict(r) for r in rows]), 200
 
 
@app.route("/applicants/<int:applicant_id>", methods=["GET"])

def get_applicant(applicant_id: int):

    db = get_db()

    row = db.execute("SELECT * FROM applicants WHERE id = ?", (applicant_id,)).fetchone()

    if not row:

        return jsonify({"error": "Applicant not found"}), 404

    return jsonify(row_to_dict(row)), 200
 
 
@app.route("/applicants", methods=["POST"])

def add_applicant():

    payload = request.get_json(silent=True) or {}

    valid, err = validate_payload(payload, required=True)

    if not valid:

        return jsonify({"error": err}), 400
 
    db = get_db()

    cur = db.cursor()

    cur.execute(

        """

        INSERT INTO applicants

        (Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,

         LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Prediction)

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """,

        (

            payload["Married"],

            payload["Education"],

            payload["Self_Employed"],

            payload["ApplicantIncome"],

            payload["CoapplicantIncome"],

            payload["LoanAmount"],

            payload["Loan_Amount_Term"],

            payload["Credit_History"],

            payload["Property_Area"],

            payload["Prediction"],

        ),

    )

    db.commit()

    new_id = cur.lastrowid
 
    row = db.execute("SELECT * FROM applicants WHERE id = ?", (new_id,)).fetchone()

    return jsonify(row_to_dict(row)), 201
 
 
@app.route("/applicants/<int:applicant_id>", methods=["PUT"])

def update_applicant(applicant_id: int):

    """

    Full or partial update by specifying:

    {

        "column": "LoanAmount",

        "new_value": 150

    }

    OR send multiple fields with "data": { ... } to update several columns:

    {

        "data": { "LoanAmount": 150, "Prediction": 1 }

    }

    """

    payload = request.get_json(silent=True) or {}
 
    db = get_db()

    # Support single-field update

    if "column" in payload and "new_value" in payload:

        column = payload["column"]

        if column not in UPDATABLE_COLUMNS:

            return jsonify({"error": f"Column '{column}' is not updatable or does not exist."}), 400
 
        # Basic type validation on-the-fly for integers

        if column in {"ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Prediction"}:

            if not isinstance(payload["new_value"], int):

                return jsonify({"error": f"Field '{column}' must be an integer."}), 400
 
        sql = f"UPDATE applicants SET {column} = ? WHERE id = ?"

        cur = db.execute(sql, (payload["new_value"], applicant_id))

        db.commit()

        if cur.rowcount == 0:

            return jsonify({"error": "Applicant not found"}), 404

        row = db.execute("SELECT * FROM applicants WHERE id = ?", (applicant_id,)).fetchone()

        return jsonify(row_to_dict(row)), 200
 
    # Support multi-field update

    if "data" in payload and isinstance(payload["data"], dict):

        updates = []

        values = []

        for col, val in payload["data"].items():

            if col not in UPDATABLE_COLUMNS:

                return jsonify({"error": f"Column '{col}' is not updatable or does not exist."}), 400

            # Inline type checks

            if col in {"ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Prediction"}:

                if not isinstance(val, int):

                    return jsonify({"error": f"Field '{col}' must be an integer."}), 400

            updates.append(f"{col} = ?")

            values.append(val)
 
        if not updates:

            return jsonify({"error": "No valid fields provided for update."}), 400
 
        values.append(applicant_id)

        sql = f"UPDATE applicants SET {', '.join(updates)} WHERE id = ?"

        cur = db.execute(sql, values)

        db.commit()

        if cur.rowcount == 0:

            return jsonify({"error": "Applicant not found"}), 404

        row = db.execute("SELECT * FROM applicants WHERE id = ?", (applicant_id,)).fetchone()

        return jsonify(row_to_dict(row)), 200
 
    return jsonify({"error": "Provide either 'column' and 'new_value', or 'data' object."}), 400
 
 
@app.route("/applicants/<int:applicant_id>", methods=["DELETE"])

def delete_applicant(applicant_id: int):

    db = get_db()

    cur = db.execute("DELETE FROM applicants WHERE id = ?", (applicant_id,))

    db.commit()

    if cur.rowcount == 0:

        return jsonify({"error": "Applicant not found"}), 404

    return jsonify({"status": "deleted", "id": applicant_id}), 200
 
 
# ---------- Entrypoint ----------
 
if __name__ == "__main__":

    # For local dev

    app.run(host="0.0.0.0", port=5000, debug=True)
 