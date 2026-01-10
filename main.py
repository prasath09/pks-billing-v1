
# main.py — Steel Billing + Inventory (MySQL-backed)
# DEPLOY TRIGGER - Jan 2026

import os
import os

from dotenv import load_dotenv

import io
import time
from datetime import datetime, date
from typing import List, Optional,Any
from decimal import Decimal

import pandas as pd
import mysql.connector

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field



# ============================== Config ==============================
# Load DB config from environment variables


ITEM_COLUMNS = ["ITEM NAME", "GST %", "UNIT", "HSN CODE", "PRICE", "STOCK QTY", "ID"]
PURCHASE_COLUMNS = ["ITEM NAME", "QTY", "GST %", "PRICE", "DATE", "BILL NO", "ROW ID"]

# ============================== DB helpers ==============================



# ============================== DB helpers ==============================


DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

missing = [k for k, v in {
    "DB_HOST": DB_HOST,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_NAME": DB_NAME,
}.items() if not v]

if missing:
    raise RuntimeError(f"Missing required DB env vars: {', '.join(missing)}")

def _get_connection(db: bool = True):
    kwargs = {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
    }

    if db:
        kwargs["database"] = DB_NAME

    return mysql.connector.connect(**kwargs)
print(
    f"DB CONNECT → host={DB_HOST}, port={DB_PORT}, db={DB_NAME}"
)

def _init_db():
    """
    Just ensure the database is reachable.
    Does NOT create or alter any tables.
    """
    try:
        conn = _get_connection(db=True)
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Database '{DB_NAME}' not reachable: {e}")


# ============================== Utils ==============================

@app.get("/")
def root():
    return {
        "status": "OK",
        "message": "PKS Billing Backend Running"
    }

def _norm_name(name: str) -> str:
    return (name or "").strip().lower()


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN check
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def _normalize_date_str(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return s
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s


def _item_db_row_to_api(row: dict) -> dict:
    """Map DB row -> API dict with Excel-style keys."""
    return {
        "ITEM NAME": _safe_str(row.get("item_name")),
        "GST %": _safe_float(row.get("gst_percent"), 0.0),
        "UNIT": _safe_str(row.get("unit") or "KGS"),
        "HSN CODE": _safe_str(row.get("hsn_code")),
        "PRICE": _safe_float(row.get("price"), 0.0),
        "STOCK QTY": _safe_float(row.get("stock_qty"), 0.0),
        "ID": _safe_str(row.get("ext_id")),
    }


def _purchase_db_row_to_api(row: dict) -> dict:
    d = row.get("date")
    if d is None:
        date_str = ""
    else:
        # mysql returns date as datetime.date or datetime
        try:
            date_str = d.isoformat()
        except Exception:
            date_str = _safe_str(d)
    return {
        "ITEM NAME": _safe_str(row.get("item_name")),
        "QTY": _safe_float(row.get("qty"), 0.0),
        "GST %": _safe_float(row.get("gst_percent"), 0.0),
        "PRICE": _safe_float(row.get("price"), 0.0),
        "DATE": date_str,
        "BILL NO": _safe_str(row.get("bill_no")),
        "ROW ID": _safe_str(row.get("row_id")),
    }


def _get_item_by_name_raw(name: str) -> Optional[dict]:
    """Return DB row for item by name (case-insensitive) or None."""
    key = _norm_name(name)
    if not key:
        return None
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty "
        "FROM items_s WHERE LOWER(item_name) = %s",
        (key,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def _get_all_items_raw() -> List[dict]:
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty "
        "FROM items_s ORDER BY item_name ASC"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _null_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _get_purchases_filtered_raw(
    item: str = "",
    bill_no: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 500,
) -> List[dict]:
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    clauses = []
    params = []

    if item:
        clauses.append("LOWER(item_name) = %s")
        params.append(_norm_name(item))

    if bill_no:
        clauses.append("LOWER(bill_no) LIKE %s")
        params.append("%" + bill_no.strip().lower() + "%")

    if date_from:
        df = _normalize_date_str(date_from)
        clauses.append("date >= %s")
        params.append(df)

    if date_to:
        dt = _normalize_date_str(date_to)
        clauses.append("date <= %s")
        params.append(dt)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT item_name, qty, gst_percent, price, date, bill_no, row_id
        FROM purchases_s
        {where}
        ORDER BY id DESC
        LIMIT %s
    """
    params.append(limit)

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _nullish(s):
    if s in ("", None, "null", "None"):
        return None
    return s


def _to_date(v):
    """
    Accepts: None, 'YYYY-MM-DD', date object.
    Returns: date or None
    """
    if v in (None, "", "null", "None"):
        return None
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


# ============================== FastAPI app ==============================

app = FastAPI(title="Inventory MySQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://billing-pks-production.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    _init_db()


# ---------- Health ----------

@app.get("/health")
def health():
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        cur.close()
        conn.close()
        return {"ok": True, "db": DB_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ============================== STOCKS (MySQL) ==============================

STOCK_TABLE = "stock_report"


def _find_stock_header_row(raw: pd.DataFrame) -> int:
    """
    Find the header row by scanning first ~20 rows for 'Description' and 'HSN'
    """
    max_scan = min(20, len(raw))
    for i in range(max_scan):
        row = raw.iloc[i].astype(str).tolist()
        text = " ".join(row).lower()
        if "description" in text and "hsn" in text:
            return i
    return -1


def _stock_rename_map():
    return {
        "Description": "description",
        "Op.Qty": "op_qty",
        "Op.Val": "op_val",
        "Pr.Qty": "pr_qty",
        "Pr.Val": "pr_val",
        "Prod": "prod",
        "Total": "total",
        "Sale.Qty": "sale_qty",
        "Sale.Val": "sale_val",
        "Cons": "cons",
        "Cl.Bal": "cl_bal",
        "Cl.Val": "cl_val",
        "Batch ID": "batch_id",
        "Absolute Position": "absolute_position",
        "Unit": "unit",
        "Barcode": "barcode",
        "Min Level": "min_level",
        "Reorder Level": "reorder_level",
        "Max Level": "max_level",
        "Short Name": "short_name",
        "HSN Code": "hsn_code",
    }


def _parse_stock_excel_bytes(xbytes: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(xbytes), header=None)

    hdr = _find_stock_header_row(raw)
    if hdr < 0:
        raise HTTPException(
            status_code=400,
            detail="Could not detect header row (Description/HSN not found).",
        )

    headers = raw.iloc[hdr].astype(str).str.strip().tolist()
    df = raw.iloc[hdr + 1:].copy()
    df.columns = headers

    # drop empty/nan headers
    df = df.loc[:, [c for c in df.columns if c and str(c).strip().lower() not in ("nan", "none", "")]]

    # strip column keys for safer rename
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.rename(columns=_stock_rename_map())

    if "description" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Header 'Description' not found. Found: {list(df.columns)}",
        )

    # keep only known columns
    keep = list(_stock_rename_map().values())
    df = df[[c for c in keep if c in df.columns]]

    # clean
    df["description"] = df["description"].astype(str).str.strip()
    df = df[(df["description"] != "") & (df["description"].str.lower() != "description")]

    # numeric safety
    df = df.fillna(0)

    return df


@app.get("/stocks")
def get_stocks(q: str = "", limit: int = 200, offset: int = 0):
    q = (q or "").strip()

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        params = {}
        where = ""
        if q:
            where = "WHERE description LIKE %(q)s OR hsn_code LIKE %(q)s"
            params["q"] = f"%{q}%"

        cur.execute(f"SELECT COUNT(*) AS n FROM {STOCK_TABLE} {where}", params)
        total = cur.fetchone()["n"]

        cur.execute(
            f"""
            SELECT id, description, hsn_code, unit,
                   op_qty, pr_qty, sale_qty, cl_bal, cl_val
            FROM {STOCK_TABLE}
            {where}
            ORDER BY id DESC
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            {**params, "limit": int(limit), "offset": int(offset)},
        )
        rows = cur.fetchall()
        return {"total": total, "rows": rows}
    finally:
        cur.close()
        conn.close()


@app.get("/stocks/suggest")
def suggest_stocks(q: str = "", limit: int = 12):
    q = (q or "").strip()
    if not q:
        return []

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            f"""
            SELECT description
            FROM {STOCK_TABLE}
            WHERE description LIKE %s
            GROUP BY description
            ORDER BY MAX(id) DESC
            LIMIT %s
            """,
            (f"%{q}%", int(limit)),
        )
        return [r["description"] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


@app.post("/stocks/import_excel")
async def import_stock_excel(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Upload an Excel file (.xlsx/.xls).")

    xbytes = await file.read()
    df = _parse_stock_excel_bytes(xbytes)

    cols = list(df.columns)
    col_sql = ", ".join(cols)
    ph_sql = ", ".join([f"%({c})s" for c in cols])
    sql = f"INSERT INTO {STOCK_TABLE} ({col_sql}) VALUES ({ph_sql})"

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    try:
        data = df.to_dict(orient="records")
        cur.executemany(sql, data)
        conn.commit()
        return {"inserted": len(data)}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


# ============================== API Models ==============================
class B2CReturnItem(BaseModel):
    sale_item_id: int
    return_qty: float


class B2CReturnPayload(BaseModel):
    invoice_no: str
    return_date: str
    items: List[B2CReturnItem]



class ItemPKSteelPayload(BaseModel):
    id: Optional[int] = None

    item_name: Optional[str] = None
    tax_percent: Optional[int] = None
    company: Optional[str] = None
    group_name: Optional[str] = None
    salt: Optional[str] = None
    primary_unit: Optional[str] = None
    alternate_unit: Optional[str] = None
    conversion_factor: Optional[float] = None
    price_per: Optional[str] = None
    basic_price: Optional[float] = None
    purchase_price: Optional[float] = None
    sale_price: Optional[float] = None
    mrp: Optional[float] = None
    qntls_per_nug: Optional[int] = None
    op_packs: Optional[int] = None
    op_stock_value: Optional[float] = None
    subitem1_title: Optional[str] = None
    exp_date: Optional[str] = None
    completebarcode: Optional[int] = None
    srno: Optional[str] = None
    itemid: Optional[int] = None
    subitemid: Optional[int] = None
    mfg_date: Optional[str] = None
    picture: Optional[str] = None
    alt_name: Optional[str] = None
    self_val_price: Optional[float] = None
    min_sale_price: Optional[float] = None
    min_level: Optional[int] = None
    max_level: Optional[int] = None
    reorder_level: Optional[int] = None
    op_loose: Optional[int] = None
    my_packs: Optional[int] = None
    my_loose: Optional[int] = None
    my_op_stock_val: Optional[int] = None
    hsn_code: Optional[str] = None
    sgst_percent: Optional[int] = None
    cgst_percent: Optional[int] = None
    igst_percent: Optional[int] = None
    additionaltax_percent: Optional[int] = None
    additionaltaxon: Optional[str] = None
    taxsystem: Optional[str] = None
    subitem2_title: Optional[str] = None
    subitem3_title: Optional[str] = None
    subitem4_title: Optional[str] = None
    subitem5_title: Optional[str] = None


class Item(BaseModel):
    name: str = Field(..., alias="ITEM NAME")
    gst: float = Field(0, alias="GST %")
    unit: str = Field("KGS", alias="UNIT")
    hsn: str = Field("", alias="HSN CODE")
    price: float = Field(0, alias="PRICE")
    stockQty: float = Field(0, alias="STOCK QTY")
    id: Optional[str] = Field(None, alias="ID")

    def to_row(self):
        return {
            "ITEM NAME": (self.name or "").strip(),
            "GST %": _safe_float(self.gst),
            "UNIT": self.unit or "KGS",
            "HSN CODE": self.hsn or "",
            "PRICE": _safe_float(self.price),
            "STOCK QTY": _safe_float(self.stockQty),
            "ID": self.id or "",
        }


class AddItemPayload(BaseModel):
    name: str
    hsn_code: Optional[str] = ""


class UpsertPayload(BaseModel):
    name: str
    gst: float


class UpsertFullPayload(BaseModel):
    items: List[Item]


class PurchasePayload(BaseModel):
    # REQUIRED
    item_id: int
    qty: float

    # OPTIONAL DISPLAY
    item_name: Optional[str] = None

    # PRICES
    purchase_price: Optional[float] = 0
    selling_price: Optional[float] = 0
    mrp: Optional[float] = 0
    basic_price: Optional[float] = None

    # TAX
    gst_percent: Optional[float] = 0

    # META
    purchase_state: Optional[str] = None
    bill_no: Optional[str] = None
    party_name: Optional[str] = None
    date: Optional[str] = None     

class PurchaseReturnPayload(BaseModel):   
    purchase_id: int
    return_qty: Decimal = Field(gt=0)

class AccountPayload(BaseModel):
    account_id: int
    account_name: str
    group_name: Optional[str] = None

    op_bal: Optional[float] = None
    dc_dr_cr: Optional[str] = None  # 'D' or 'C'

    address_line_1: Optional[str] = None
    address_line_2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None

    phone: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None

    gst_no: Optional[str] = None


class SaleItemPayload(BaseModel):
    item_name: str
    qty: float
    price: float
    gst: float = 0

class SaleItemPayload(BaseModel):
    item_name: str
    qty: float
    price: float
    gst: float
    hsn: Optional[str] = None
    unit: Optional[str] = "KG"


class SaleCreatePayload(BaseModel):
    # ===== EXISTING =====
    account_id: int
    bill_no: Optional[str] = None
    date: str
    party_name: str = ""

    taxable: float = 0
    discount_amount: float = 0
    discounted_taxable: float = 0
    gst_amount: float = 0
    round_off: float = 0
    grand_total: float = 0

    paid_now: float = 0
    balance: float = 0
    payment_mode: Optional[str] = None

    split_enabled: bool = False
    split1_amount: float = 0
    split1_mode: Optional[str] = None
    split1_ref: Optional[str] = None
    split2_amount: float = 0
    split2_mode: Optional[str] = None
    split2_ref: Optional[str] = None

    buyer_gstin: Optional[str] = None
    buyer_state: Optional[str] = None
    buyer_state_code: Optional[str] = None

    # ===== NEW (REQUIRED) =====
    invoice_type: str  # "B2C" or "B2B"

    # 🚚 Transport / E-way
    eway_no: Optional[str] = None
    vehicle_no: Optional[str] = None
    kms: Optional[int] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    items: List[SaleItemPayload]

class PurchaseReturnItem(BaseModel):
    purchase_id: int
    return_qty: float

class PurchaseReturnPayload(BaseModel):
    bill_no: str
    items: List[PurchaseReturnItem]

# ============================== Inventory APIs ==============================

@app.get("/inventory", response_model=List[Item])
def get_inventory():
    rows = _get_all_items_raw()
    return [_item_db_row_to_api(r) for r in rows]


@app.post("/inventory/add_item")
def add_item(payload: AddItemPayload):
    key = _norm_name(payload.name)
    if not key:
        raise HTTPException(status_code=400, detail="Empty name")

    hsn = (payload.hsn_code or "").strip()
    existing = _get_item_by_name_raw(payload.name)

    conn = _get_connection()
    conn.autocommit = True
    cur = conn.cursor(dictionary=True)

    try:
        if existing:
            if hsn:
                cur.execute("UPDATE items_s SET hsn_code = %s WHERE id = %s", (hsn, existing["id"]))
                existing["hsn_code"] = hsn
            return _item_db_row_to_api(existing)

        cur.execute(
            """
            INSERT INTO items_s (item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (payload.name.strip(), 0.0, "KGS", hsn, 0.0, 0.0, ""),
        )
        new_id = cur.lastrowid
        cur.execute(
            "SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id FROM items_s WHERE id = %s",
            (new_id,),
        )
        row = cur.fetchone()
        return _item_db_row_to_api(row)
    finally:
        cur.close()
        conn.close()


@app.post("/inventory/upsert_gst", response_model=Item)
def upsert_gst(payload: UpsertPayload):
    key = _norm_name(payload.name)
    if not key:
        raise HTTPException(status_code=400, detail="Empty name")

    existing = _get_item_by_name_raw(payload.name)
    conn = _get_connection()
    conn.autocommit = True
    cur = conn.cursor(dictionary=True)

    try:
        if existing:
            cur.execute(
                "UPDATE items_s SET gst_percent = %s WHERE id = %s",
                (_safe_float(payload.gst), existing["id"]),
            )
            existing["gst_percent"] = _safe_float(payload.gst)
            row = existing
        else:
            cur.execute(
                """
                INSERT INTO items_s (item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (payload.name.strip(), _safe_float(payload.gst), "KGS", "", 0.0, 0.0, ""),
            )
            new_id = cur.lastrowid
            cur.execute(
                "SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id FROM items_s WHERE id = %s",
                (new_id,),
            )
            row = cur.fetchone()

        return _item_db_row_to_api(row)
    finally:
        cur.close()
        conn.close()


@app.post("/inventory/upsert_bulk", response_model=List[Item])
def upsert_bulk(payload: UpsertFullPayload):
    conn = _get_connection()
    conn.autocommit = True
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute(
            "SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id FROM items_s"
        )
        existing_rows = cur.fetchall()
        by_name = {row["item_name"].lower(): row for row in existing_rows}

        for it in payload.items:
            key = _norm_name(it.name)
            if not key:
                continue

            if key in by_name:
                row = by_name[key]
                gst = _safe_float(it.gst)
                unit = it.unit or row["unit"] or "KGS"
                hsn = it.hsn if it.hsn is not None else row["hsn_code"]
                price = _safe_float(it.price if it.price is not None else row["price"])
                stock = _safe_float(it.stockQty if it.stockQty is not None else row["stock_qty"])
                cur.execute(
                    """
                    UPDATE items_s
                    SET gst_percent = %s, unit = %s, hsn_code = %s, price = %s, stock_qty = %s
                    WHERE id = %s
                    """,
                    (gst, unit, hsn, price, stock, row["id"]),
                )
            else:
                data = it.to_row()
                cur.execute(
                    """
                    INSERT INTO items_s (item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        data["ITEM NAME"],
                        data["GST %"],
                        data["UNIT"],
                        data["HSN CODE"],
                        data["PRICE"],
                        data["STOCK QTY"],
                        data["ID"],
                    ),
                )

        cur.execute(
            "SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty, ext_id "
            "FROM items_s ORDER BY item_name ASC"
        )
        rows = cur.fetchall()
        return [_item_db_row_to_api(r) for r in rows]
    finally:
        cur.close()
        conn.close()


@app.get("/inventory/suggest")
def suggest(q: str = Query("", description="partial item name"), limit: int = 20):
    qn = (q or "").strip().lower()

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        if qn:
            like = "%" + qn + "%"
            prefix = qn + "%"
            cur.execute(
                """
                SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty
                FROM items_s
                WHERE LOWER(item_name) LIKE %s
                ORDER BY (LOWER(item_name) LIKE %s) DESC, item_name ASC
                LIMIT %s
                """,
                (like, prefix, int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT id, item_name, gst_percent, unit, hsn_code, price, stock_qty
                FROM items_s
                ORDER BY item_name ASC
                LIMIT %s
                """,
                (int(limit),),
            )

        rows = cur.fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "id": r.get("id"),
                    "name": _safe_str(r.get("item_name")),
                    "gst": _safe_float(r.get("gst_percent"), 0.0),
                    "unit": _safe_str(r.get("unit") or "KG"),
                    "hsn": _safe_str(r.get("hsn_code")),
                    "price": _safe_float(r.get("price"), 0.0),
                    "stockQty": _safe_float(r.get("stock_qty"), 0.0),
                }
            )
        return out
    finally:
        cur.close()
        conn.close()


# ---------- Purchases ----------


@app.post("/inventory/add_purchase")
def add_purchase(p: PurchasePayload):

    print("\n===== ADD PURCHASE PAYLOAD RECEIVED =====")
    print(p.dict())
    print("========================================\n")

    # ---------- BASIC VALIDATION ----------
    if not p.item_id:
        raise HTTPException(status_code=400, detail="item_id is required")

    qty_val = _safe_float(p.qty)
    if qty_val <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    try:
        # ---------- FETCH ITEM ----------
        cur.execute(
            """
            SELECT id, item_name, stock_qty, primary_unit
            FROM items_s
            WHERE id = %s
            """,
            (p.item_id,),
        )
        item = cur.fetchone()

        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        # ---------- UPDATE STOCK ----------
        new_stock = _safe_float(item["stock_qty"]) + qty_val
        cur.execute(
            "UPDATE items_s SET stock_qty = %s WHERE id = %s",
            (new_stock, item["id"]),
        )

        # ---------- NORMALIZE NUMBERS ----------
        purchase_price = _safe_float(p.purchase_price)
        selling_price = _safe_float(p.selling_price)
        gst_percent = _safe_float(p.gst_percent)
        mrp = _safe_float(p.mrp)
        basic_price = _safe_float(p.basic_price) if p.basic_price else None

        # ---------- NORMALIZE STRINGS ----------
        purchase_state = (p.purchase_state or "").strip() or None
        party_name = (p.party_name or "").strip() or None
        bill_no = (p.bill_no or "").strip() or None

        # ---------- UNIT (SAFE DEFAULT) ----------
        primary_unit = item.get("primary_unit") or "KGS"

        # ---------- DATE ----------
        norm_date = _normalize_date_str(p.date)
        dt_str = (
            f"{norm_date} {datetime.now().strftime('%H:%M:00')}"
            if norm_date
            else None
        )

        # ---------- INSERT PURCHASE ----------
        cur.execute(
            """
            INSERT INTO purchases_s (
                item_id,
                item_name,
                qty,
                remaining_qty,
                purchase_price,
                selling_price,
                purchase_state,
                gst_percent,
                date,
                bill_no,
                party_name,
                primary_unit,
                basic_price,
                mrp
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                item["id"],
                item["item_name"],
                qty_val,
                qty_val,
                purchase_price,
                selling_price,
                purchase_state,
                gst_percent,
                dt_str,
                bill_no,
                party_name,
                primary_unit,
                basic_price,
                mrp,
            ),
        )

        conn.commit()

        return {
            "ok": True,
            "item_id": item["id"],
            "item_name": item["item_name"],
            "added_qty": qty_val,
            "updated_stock": new_stock,
        }

    except Exception as e:
        conn.rollback()
        print("🔥 ADD PURCHASE ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()
@app.get("/purchases/suggest")
def purchases_suggest(q: str = Query("", description="partial item name")):
    print("🔥 /purchases/suggest API HIT")
    print("Query received:", q)
    qn = (q or "").strip().lower()
    

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    base_sql = """
        SELECT
            p.item_name,
            p.gst_percent     AS last_gst,
            p.selling_price   AS last_price,
            i.unit,
            i.hsn_code,
            i.gst_percent     AS item_gst,
            i.price           AS item_price,
            i.stock_qty
        FROM purchases_s p
        JOIN (
            SELECT item_id, MAX(date) AS max_date
            FROM purchases_s
            GROUP BY item_id
        ) lp ON lp.item_id = p.item_id AND lp.max_date = p.date
        JOIN items_s i ON i.id = p.item_id
    """

    params = []
    if qn:
        base_sql += " WHERE LOWER(p.item_name) LIKE %s"
        params.append(f"%{qn}%")

    base_sql += " ORDER BY p.item_name ASC"

    try:
        cur.execute(base_sql, tuple(params))
        rows = cur.fetchall()

        out = []
        for r in rows:
            price = _safe_float(r.get("last_price"), _safe_float(r.get("item_price"), 0.0))
            gst = _safe_float(r.get("last_gst"), _safe_float(r.get("item_gst"), 0.0))
            out.append({
                "name": _safe_str(r.get("item_name")),
                "unit": "KG",
                "hsn": _safe_str(r.get("hsn_code")),
                "gst": gst,
                "price": price,
                "last_gst": _safe_float(r.get("last_gst"), 0.0),
                "last_price": _safe_float(r.get("last_price"), 0.0),
                "stockQty": _safe_float(r.get("stock_qty"), 0.0),
            })
        return out
    finally:
        cur.close()
        conn.close()




@app.get("/purchases/last")
def purchases_last(name: str):
    """
    Return Rate and GST% from latest Purchases row.
    Falls back to Items table if Purchases has no rows for that item.
    """
    key = _norm_name(name)
    if not key:
        raise HTTPException(status_code=400, detail="Item name required")

    item_row = _get_item_by_name_raw(name)
    if not item_row:
        raise HTTPException(status_code=404, detail="Item not found in Items table")

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute(
            """
            SELECT gst_percent, selling_price
            FROM purchases_s
            WHERE item_id = %s
            ORDER BY date DESC, id DESC
            LIMIT 1
            """,
            (item_row["id"],),
        )
        last = cur.fetchone()

        if last:
            rate = _safe_float(last.get("selling_price"), _safe_float(item_row.get("price"), 0.0))
            gst = _safe_float(last.get("gst_percent"), _safe_float(item_row.get("gst_percent"), 0.0))
        else:
            rate = _safe_float(item_row.get("price"), 0.0)
            gst = _safe_float(item_row.get("gst_percent"), 0.0)

        return {
            "name": _safe_str(item_row.get("item_name")),
            "unit": "KG",
            "hsn": _safe_str(item_row.get("hsn_code")),
            "price": rate,
            "gst": gst,
            "stockQty": _safe_float(item_row.get("stock_qty"), 0.0),
        }
    finally:
        cur.close()
        conn.close()


# ---------- Sales FIFO ----------

def _normalize_invoice_type(v):
    if not v:
        return "B2C"
    v = v.upper().strip()
    if v in ("B2C", "B TO C", "BTOC"):
        return "B2C"
    if v in ("B2B", "B TO B", "BTOB"):
        return "B2B"
    return "B2C"

def _next_bill_no(cur, invoice_type):
    prefix = "PKS-C-" if invoice_type == "B2C" else "PKS-B-"

    cur.execute(
        """
        SELECT bill_no
        FROM sales
        WHERE invoice_type = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (invoice_type,),
    )
    row = cur.fetchone()

    if not row:
        return f"{prefix}001"

    last = row["bill_no"].replace(prefix, "")
    try:
        n = int(last) + 1
    except:
        n = 1

    return f"{prefix}{str(n).zfill(3)}"



@app.post("/sales/create_fifo")
def create_sale_fifo(payload: SaleCreatePayload):
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items in sale")

    norm_date = _normalize_date_str(payload.date)
    if not norm_date:
        raise HTTPException(status_code=400, detail="Invalid sale date")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    try:
        # ==================================================
        # 1. Invoice Type + Bill Number (BACKEND CONTROLLED)
        # ==================================================
        invoice_type = _normalize_invoice_type(payload.invoice_type)
        bill_no = _next_bill_no(cur, invoice_type)

        print("Invoice Type:", invoice_type)
        print("Bill No:", bill_no)
        print("E-Way:", payload.eway_no)
        print("Vehicle:", payload.vehicle_no)
        print("KMs:", payload.kms)
        print("From:", payload.from_date, "To:", payload.to_date)

        # ==================================================
        # 2. Insert SALES header
        # ==================================================
        cur.execute(
            """
            INSERT INTO sales (
                bill_no, invoice_type, date,
                account_id, party_name,

                grand_total, taxable, discount_amount,
                discounted_taxable, gst_amount, round_off,

                paid_now, balance, payment_mode,

                split_enabled,
                split1_amount, split1_mode, split1_ref,
                split2_amount, split2_mode, split2_ref,

                buyer_gstin, buyer_state, buyer_state_code,

                eway_no, vehicle_no, kms, from_date, to_date,

                company_id
            )
            VALUES (
                %s,%s,%s,
                %s,%s,

                %s,%s,%s,
                %s,%s,%s,

                %s,%s,%s,

                %s,
                %s,%s,%s,
                %s,%s,%s,

                %s,%s,%s,

                %s,%s,%s,%s,%s,

                %s
            )
            """,
            (
                bill_no,
                invoice_type,
                norm_date,

                payload.account_id,
                payload.party_name or "",

                payload.grand_total,
                payload.taxable,
                payload.discount_amount,
                payload.discounted_taxable,
                payload.gst_amount,
                payload.round_off,

                payload.paid_now,
                payload.balance,
                _null(payload.payment_mode),

                1 if payload.split_enabled else 0,
                payload.split1_amount,
                _null(payload.split1_mode),
                _null(payload.split1_ref),
                payload.split2_amount,
                _null(payload.split2_mode),
                _null(payload.split2_ref),

                _null(payload.buyer_gstin),
                _null(payload.buyer_state),
                _null(payload.buyer_state_code),

                _null(payload.eway_no),
                _null(payload.vehicle_no),
                payload.kms,
                _null(payload.from_date),
                _null(payload.to_date),

                1,  # company_id
            ),
        )

        sale_id = cur.lastrowid

        # ==================================================
        # 3. SALE ITEMS + FIFO STOCK CONSUMPTION
        # ==================================================
        row_no = 1

        for line in payload.items:
            item_name = line.item_name.strip()
            qty = float(line.qty)
            rate = float(line.price)
            gst = float(line.gst)

            if not item_name or qty <= 0:
                continue

            # Fetch item
            cur.execute(
                """
                SELECT id, item_name, hsn_code, unit, stock_qty
                FROM items_s
                WHERE LOWER(item_name) = %s
                LIMIT 1
                """,
                (item_name.lower(),),
            )
            item = cur.fetchone()
            if not item:
                raise HTTPException(status_code=404, detail=f"Item not found: {item_name}")

            if item["stock_qty"] < qty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient stock for {item_name}",
                )

            line_amount = qty * rate

            # Insert sale_items
            cur.execute(
                """
                INSERT INTO sale_items (
                    sale_id, item_id, row_no,
                    item_name, hsn_code, unit,
                    qty, sale_rate, gst_percent, line_amount
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    sale_id,
                    item["id"],
                    row_no,
                    item["item_name"],
                    item["hsn_code"],
                    item["unit"] or "KG",
                    qty,
                    rate,
                    gst,
                    line_amount,
                ),
            )

            sale_item_id = cur.lastrowid
            row_no += 1

            # FIFO batches
            cur.execute(
                """
                SELECT id, remaining_qty
                FROM purchases_s
                WHERE item_id = %s AND remaining_qty > 0
                ORDER BY date ASC, id ASC
                FOR UPDATE
                """,
                (item["id"],),
            )

            remaining = qty
            for batch in cur.fetchall():
                if remaining <= 0:
                    break

                use_qty = min(batch["remaining_qty"], remaining)

                cur.execute(
                    """
                    INSERT INTO sale_lines
                    (sale_id, sale_item_id, item_id, purchase_id,
                     qty_sold, sale_rate, gst_percent)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        sale_id,
                        sale_item_id,
                        item["id"],
                        batch["id"],
                        use_qty,
                        rate,
                        gst,
                    ),
                )

                cur.execute(
                    """
                    UPDATE purchases_s
                    SET remaining_qty = remaining_qty - %s
                    WHERE id = %s
                    """,
                    (use_qty, batch["id"]),
                )

                remaining -= use_qty

            # Update item stock
            cur.execute(
                """
                UPDATE items_s
                SET stock_qty = stock_qty - %s
                WHERE id = %s
                """,
                (qty, item["id"]),
            )

        conn.commit()

        return {
            "ok": True,
            "sale_id": sale_id,
            "bill_no": bill_no,
            "invoice_type": invoice_type,
            "message": "Sale created successfully with FIFO",
        }

    except Exception as e:
        conn.rollback()
        print("DB ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()


@app.post("/sales/create_fifo01")
def create_sale_fifo(payload: SaleCreatePayload):
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items in sale")

    norm_date = _normalize_date_str(payload.date)
    if not norm_date:
        raise HTTPException(status_code=400, detail="Invalid sale date")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

  

    party_name = _null(payload.party_name)
    payment_mode = _null(payload.payment_mode)
    split1_mode = _null(payload.split1_mode)
    split1_ref = _null(payload.split1_ref)
    split2_mode = _null(payload.split2_mode)
    split2_ref = _null(payload.split2_ref)
    buyer_gstin = _null(payload.buyer_gstin)
    buyer_state = _null(payload.buyer_state)
    buyer_state_code = _null(payload.buyer_state_code)

    try:
        # =========================================================
        # BLOCK 1: SALES HEADER (schema-aligned, SAFE)
        # =========================================================
        cur.execute(
            """
            INSERT INTO sales (
                bill_no,
                date,
                account_id,
                party_name,
                grand_total,
                taxable,
                discount_amount,
                discounted_taxable,
                gst_amount,
                round_off,
                paid_now,
                balance,
                payment_mode,
                split_enabled,
                split1_amount,
                split1_mode,
                split1_ref,
                split2_amount,
                split2_mode,
                split2_ref,
                buyer_gstin,
                buyer_state,
                buyer_state_code,
                company_id
            )
            VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s
            )
            """,
            (
                payload.bill_no,                 # bill_no
                norm_date,                       # date
                payload.account_id,              # account_id
                party_name or "",

                _safe_float(payload.grand_total),
                _safe_float(payload.taxable),
                _safe_float(payload.discount_amount),
                _safe_float(payload.discounted_taxable),
                _safe_float(payload.gst_amount),
                _safe_float(payload.round_off),

                _safe_float(payload.paid_now),
                _safe_float(payload.balance),
                payment_mode,
                1 if payload.split_enabled else 0,

                _safe_float(payload.split1_amount),
                split1_mode,
                split1_ref,

                _safe_float(payload.split2_amount),
                split2_mode,
                split2_ref,

                buyer_gstin,
                buyer_state,
                buyer_state_code,

                1,  # company_id (explicit)
            ),
        )

        sale_id = cur.lastrowid

        # =========================================================
        # BLOCK 2: SALE ITEMS + FIFO CONSUMPTION
        # =========================================================
        row_no = 1
        running_total = 0.0

        for line in payload.items:
            item_name = _safe_str(getattr(line, "item_name", "")).strip()
            qty_to_sell = _safe_float(getattr(line, "qty", 0.0))
            sale_rate = _safe_float(getattr(line, "price", 0.0))
            gst_percent = _safe_float(getattr(line, "gst", 0.0))

            if not item_name or qty_to_sell <= 0:
                row_no += 1
                continue

            # Fetch item
            cur.execute(
                """
                SELECT id, item_name, hsn_code, unit, stock_qty
                FROM items_s
                WHERE LOWER(item_name) = %s
                LIMIT 1
                """,
                (_norm_name(item_name),),
            )
            item = cur.fetchone()
            if not item:
                raise HTTPException(status_code=404, detail=f"Item not found: {item_name}")

            if _safe_float(item["stock_qty"]) < qty_to_sell:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient stock for {item_name}",
                )

            line_amount = qty_to_sell * sale_rate
            running_total += line_amount

            # Insert sale_items
            cur.execute(
                """
                INSERT INTO sale_items (
                    sale_id,
                    item_id,
                    row_no,
                    item_name,
                    hsn_code,
                    unit,
                    qty,
                    sale_rate,
                    gst_percent,
                    line_amount
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    sale_id,
                    item["id"],
                    row_no,
                    item["item_name"],
                    item["hsn_code"],
                    item["unit"] or "KG",
                    qty_to_sell,
                    sale_rate,
                    gst_percent,
                    line_amount,
                ),
            )
            sale_item_id = cur.lastrowid
            row_no += 1

            # FIFO batches
            cur.execute(
                """
                SELECT id, remaining_qty
                FROM purchases_s
                WHERE item_id = %s AND remaining_qty > 0
                ORDER BY date ASC, id ASC
                FOR UPDATE
                """,
                (item["id"],),
            )
            batches = cur.fetchall()

            remaining = qty_to_sell

            for batch in batches:
                if remaining <= 0:
                    break

                batch_qty = _safe_float(batch["remaining_qty"])
                use_qty = min(batch_qty, remaining)

                cur.execute(
                    """
                    INSERT INTO sale_lines
                        (sale_id, sale_item_id, item_id, purchase_id, qty_sold, sale_rate, gst_percent)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        sale_id,
                        sale_item_id,
                        item["id"],
                        batch["id"],
                        use_qty,
                        sale_rate,
                        gst_percent,
                    ),
                )

                cur.execute(
                    """
                    UPDATE purchases_s
                    SET remaining_qty = remaining_qty - %s
                    WHERE id = %s
                    """,
                    (use_qty, batch["id"]),
                )

                remaining -= use_qty

            # Safety fallback (should not happen)
            if remaining > 0:
                cur.execute(
                    """
                    INSERT INTO sale_lines
                        (sale_id, sale_item_id, item_id, purchase_id, qty_sold, sale_rate, gst_percent)
                    VALUES (%s,%s,%s,NULL,%s,%s,%s)
                    """,
                    (
                        sale_id,
                        sale_item_id,
                        item["id"],
                        remaining,
                        sale_rate,
                        gst_percent,
                    ),
                )

            # Update item stock
            cur.execute(
                """
                UPDATE items_s
                SET stock_qty = stock_qty - %s
                WHERE id = %s
                """,
                (qty_to_sell, item["id"]),
            )

        conn.commit()

        return {
            "ok": True,
            "sale_id": sale_id,
            "bill_no": payload.bill_no,
            "grand_total": _safe_float(payload.grand_total) or running_total,
            "message": "Sale created successfully with FIFO stock adjustment.",
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"DB error while creating sale: {e}")
    finally:
        cur.close()
        conn.close()


# =======================
# GLOBAL HELPERS (REQUIRED)
# =======================

def _null(x):
    return x if x not in ("", None, "null", "None") else None


@app.post("/sales/create_fifo00")
def create_sale_fifo(payload: SaleCreatePayload):
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items in sale")

    norm_date = _normalize_date_str(payload.date)
    if not norm_date:
        raise HTTPException(status_code=400, detail="Invalid sale date")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    party_name = _null(payload.party_name)
    payment_mode = _null(payload.payment_mode)
    split1_mode = _null(payload.split1_mode)
    split1_ref = _null(payload.split1_ref)
    split2_mode = _null(payload.split2_mode)
    split2_ref = _null(payload.split2_ref)
    buyer_gstin = _null(payload.buyer_gstin)
    buyer_state = _null(payload.buyer_state)
    buyer_state_code = _null(payload.buyer_state_code)
    eway_no = _null(payload.eway_no)
    vehicle_no = _null(payload.vehicle_no)
    kms = payload.kms if payload.kms not in ("", None) else None
    from_date = _null(payload.from_date)
    to_date = _null(payload.to_date)


    try:
        # BLOCK 1: sales header
        cur.execute(
            """
            INSERT INTO sales (
                bill_no,
                date,
                account_id,
                party_name,
                taxable,
                discount_amount,
                discounted_taxable,
                gst_amount,
                round_off,
                grand_total,
                paid_now,
                balance,
                payment_mode,
                split_enabled,
                split1_amount,
                split1_mode,
                split1_ref,
                split2_amount,
                split2_mode,
                split2_ref,
                buyer_gstin,
                buyer_state,
                buyer_state_code
                eway_no,
                vehicle_no,
                kms,
                from_date,
                to_date,
            )
            VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s
            )
            """,
            (
                payload.account_id,
                payload.bill_no,
                norm_date,
                party_name or "",
                _safe_float(payload.taxable),
                _safe_float(payload.discount_amount),
                _safe_float(payload.discounted_taxable),
                _safe_float(payload.gst_amount),
                _safe_float(payload.round_off),
                _safe_float(payload.grand_total),
                _safe_float(payload.paid_now),
                _safe_float(payload.balance),
                payment_mode,
                1 if payload.split_enabled else 0,
                _safe_float(payload.split1_amount),
                split1_mode,
                split1_ref,
                _safe_float(payload.split2_amount),
                split2_mode,
                split2_ref,
                buyer_gstin,
                buyer_state,
                buyer_state_code,
                eway_no,
                vehicle_no,
                kms,
                from_date,
                to_date,
            ),
        )
        sale_id = cur.lastrowid

        # BLOCK 2: sale_items + FIFO
        running_total_amount = 0.0
        row_no = 1

        for line in payload.items:
            item_name = _safe_str(getattr(line, "item_name", "")).strip()
            qty_to_sell = _safe_float(getattr(line, "qty", 0.0))
            sale_rate = _safe_float(getattr(line, "price", 0.0))
            gst_percent = _safe_float(getattr(line, "gst", 0.0))

            if not item_name or qty_to_sell <= 0:
                row_no += 1
                continue

            cur.execute(
                """
                SELECT id, item_name, hsn_code, unit, stock_qty
                FROM items_s
                WHERE LOWER(item_name) = %s
                LIMIT 1
                """,
                (_norm_name(item_name),),
            )
            item_row = cur.fetchone()
            if not item_row:
                raise HTTPException(status_code=404, detail=f"Item not found: {item_name}")

            item_id = item_row["id"]
            current_stock = _safe_float(item_row.get("stock_qty"), 0.0)

            if current_stock < qty_to_sell:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient stock for {item_name}: have {current_stock}, need {qty_to_sell}",
                )

            line_amount = qty_to_sell * sale_rate
            running_total_amount += line_amount

            cur.execute(
                """
                INSERT INTO sale_items (
                    sale_id,
                    item_id,
                    row_no,
                    item_name,
                    hsn_code,
                    unit,
                    qty,
                    sale_rate,
                    gst_percent,
                    line_amount
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    sale_id,
                    item_id,
                    row_no,
                    _safe_str(item_row.get("item_name")),
                    _safe_str(item_row.get("hsn_code")),
                    _safe_str(item_row.get("unit") or "KG"),
                    qty_to_sell,
                    sale_rate,
                    gst_percent,
                    line_amount,
                ),
            )
            sale_item_id = cur.lastrowid
            row_no += 1

            cur.execute(
                """
                SELECT id, remaining_qty, date
                FROM purchases_s    
                WHERE item_id = %s AND remaining_qty > 0
                ORDER BY date ASC, id ASC
                FOR UPDATE
                """,
                (item_id,),
            )
            batches = cur.fetchall()

            remaining = qty_to_sell

            for batch in batches:
                if remaining <= 0:
                    break

                batch_rem = _safe_float(batch.get("remaining_qty"), 0.0)
                if batch_rem <= 0:
                    continue

                use_qty = min(batch_rem, remaining)

                cur.execute(
                    """
                    INSERT INTO sale_lines
                        (sale_id, sale_item_id, item_id, purchase_id, qty_sold, sale_rate, gst_percent)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (sale_id, sale_item_id, item_id, batch["id"], use_qty, sale_rate, gst_percent),
                )

                cur.execute(
                    """
                    UPDATE purchases_s
                    SET remaining_qty = remaining_qty - %s
                    WHERE id = %s
                    """,
                    (use_qty, batch["id"]),
                )

                remaining -= use_qty

            if remaining > 0:
                cur.execute(
                    """
                    INSERT INTO sale_lines
                        (sale_id, sale_item_id, item_id, purchase_id, qty_sold, sale_rate, gst_percent)
                    VALUES (%s,%s,%s,NULL,%s,%s,%s)
                    """,
                    (sale_id, sale_item_id, item_id, remaining, sale_rate, gst_percent),
                )
                remaining = 0

            cur.execute(
                """
                UPDATE items_s
                SET stock_qty = stock_qty - %s
                WHERE id = %s
                """,
                (qty_to_sell, item_id),
            )

        payload_grand = _safe_float(getattr(payload, "grand_total", 0.0))
        final_grand = payload_grand if payload_grand > 0 else running_total_amount

        cur.execute("UPDATE sales SET grand_total = %s WHERE id = %s", (final_grand, sale_id))

        conn.commit()

        return {
            "ok": True,
            "sale_id": sale_id,
            "bill_no": payload.bill_no,
            "grand_total": final_grand,
            "message": "Sale header + items + FIFO lines created successfully.",
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"DB error while creating sale: {e}")
    finally:
        cur.close()
        conn.close()




@app.get("/inventory/availability")
def inventory_availability(name: str = Query(""), qty: float = Query(0)):
    item_row = _get_item_by_name_raw(name)
    if not item_row:
        return {"exists": False, "stock": 0, "can_deliver": False}

    stock = _safe_float(item_row.get("stock_qty"), 0.0)
    need = _safe_float(qty, 0.0)

    can_deliver = not (need > 0 and stock < need)

    return {
        "exists": True,
        "stock": stock,
        "can_deliver": can_deliver,
        "stockQty": stock,
        "available": stock,
    }


# ---------- Accounts ----------

@app.get("/accounts")
def accounts_list(q: str = Query("", description="name/gst/mobile/phone search"), limit: int = 100):
    qn = (q or "").strip().lower()

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        if qn:
            like = f"%{qn}%"
            cur.execute(
                """
                SELECT
                  account_id, account_name, group_name,
                  gst_no, mobile, phone, email,
                  city, state, pincode,
                  address_line_1, address_line_2
                FROM accounts_list
                WHERE
                  LOWER(account_name) LIKE %s
                  OR LOWER(IFNULL(gst_no,'')) LIKE %s
                  OR LOWER(IFNULL(mobile,'')) LIKE %s
                  OR LOWER(IFNULL(phone,'')) LIKE %s
                ORDER BY account_name ASC
                LIMIT %s
                """,
                (like, like, like, like, int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT
                  account_id, account_name, group_name,
                  gst_no, mobile, phone, email,
                  city, state, pincode,
                  address_line_1, address_line_2
                FROM accounts_list
                ORDER BY account_name ASC
                LIMIT %s
                """,
                (int(limit),),
            )

        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


@app.post("/accounts/upsert")
def accounts_upsert(p: AccountPayload):
    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO accounts_list (
              account_id, account_name, group_name,
              op_bal, dc_dr_cr,
              address_line_1, address_line_2, city, state, pincode,
              phone, mobile, email, gst_no
            )
            VALUES (
              %s,%s,%s,
              %s,%s,
              %s,%s,%s,%s,%s,
              %s,%s,%s,%s
            )
            ON DUPLICATE KEY UPDATE
              account_name=VALUES(account_name),
              group_name=VALUES(group_name),
              op_bal=VALUES(op_bal),
              dc_dr_cr=VALUES(dc_dr_cr),
              address_line_1=VALUES(address_line_1),
              address_line_2=VALUES(address_line_2),
              city=VALUES(city),
              state=VALUES(state),
              pincode=VALUES(pincode),
              phone=VALUES(phone),
              mobile=VALUES(mobile),
              email=VALUES(email),
              gst_no=VALUES(gst_no)
            """,
            (
                int(p.account_id),
                _null_str(p.account_name) or "",
                _null_str(p.group_name),
                p.op_bal,
                (_null_str(p.dc_dr_cr) or None),
                _null_str(p.address_line_1),
                _null_str(p.address_line_2),
                _null_str(p.city),
                _null_str(p.state),
                _null_str(p.pincode),
                _null_str(p.phone),
                _null_str(p.mobile),
                _null_str(p.email),
                _null_str(p.gst_no),
            ),
        )

        conn.commit()
        return {"ok": True, "account_id": int(p.account_id)}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"DB error upserting account: {e}")
    finally:
        cur.close()
        conn.close()


# ---------- Items PK Steel ----------

@app.get("/items_pk_steel")
def list_items_pk_steel(q: str = "", limit: int = 300):
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        q = (q or "").strip()
        if q:
            like = f"%{q}%"
            cur.execute(
                """
                SELECT
                id,item_name,group_name,hsn_code,primary_unit,tax_percent,cgst_percent,sgst_percent,igst_percent
                FROM items_s WHERE
                item_name LIKE %s OR
                hsn_code LIKE %s OR
                group_name LIKE %s
                ORDER BY item_name
                LIMIT %s
                """,
                (like, like, like, int(limit)),
            )

        else:
            cur.execute(
                """
                SELECT
                id,item_name,group_name,hsn_code,primary_unit,tax_percent,cgst_percent,sgst_percent,igst_percent
                FROM items_s 
                ORDER BY item_name
                LIMIT %s
                """,
                (int(limit),),
            )
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


@app.get("/items_pk_steel/{item_id}")
def get_item_pk_steel(item_id: int):
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM items_s WHERE id=%s", (int(item_id),))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found")
        return row
    finally:
        cur.close()
        conn.close()


@app.post("/items_pk_steel/upsert")
def upsert_item_pk_steel(p: ItemPKSteelPayload):
    name = (p.item_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="item_name is required")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # ✅ ONLY REQUIRED COLUMNS
        cols = [
            "item_name",
            "tax_percent",
            "group_name",
            "primary_unit",
            "hsn_code",
            "sgst_percent",
            "cgst_percent",
            "igst_percent",
        ]

        values = [
            name,
            p.tax_percent,
            p.group_name,
            p.primary_unit,
            p.hsn_code,
            p.sgst_percent,
            p.cgst_percent,
            p.igst_percent,
        ]

        # ---------------- UPDATE ----------------
        if p.id:
            set_sql = ", ".join(f"{c}=%s" for c in cols)
            cur.execute(
                f"""
                UPDATE items_s
                SET {set_sql}
                WHERE id = %s
                """,
                values + [int(p.id)],
            )
            conn.commit()
            return {
                "ok": True,
                "mode": "update",
                "id": int(p.id),
            }

        # ---------------- INSERT ----------------
        placeholders = ",".join(["%s"] * len(cols))
        col_sql = ",".join(cols)

        cur.execute(
            f"""
            INSERT INTO items_s ({col_sql})
            VALUES ({placeholders})
            """,
            values,
        )

        new_id = cur.lastrowid
        conn.commit()

        return {
            "ok": True,
            "mode": "insert",
            "id": int(new_id),
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()


@app.delete("/items_pk_steel/{item_id}")
def delete_item_pk_steel(item_id: int):
    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM items_s WHERE id=%s", (int(item_id),))
        conn.commit()
        return {"ok": True, "deleted": int(item_id)}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# ---- Accounts list: mobile suggest + fetch ----
@app.get("/accounts/suggest")
def accounts_suggest(q: str = "", limit: int = 10):
    q = (q or "").strip()
    if not q:
        return []

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        # detect numeric search (account_id / mobile)
        is_numeric = q.isdigit()

        like = f"%{q}%"

        if is_numeric:
            cur.execute(
                """
                SELECT
                  account_id,
                  account_name,
                  mobile,
                  city
                FROM accounts_list
                WHERE
                  CAST(account_id AS CHAR) LIKE %s
                  OR mobile LIKE %s
                ORDER BY account_id
                LIMIT %s
                """,
                (like, like, int(limit)),
            )
        else:
            cur.execute(
                """
                SELECT
                  account_id,
                  account_name,
                  mobile,
                  city
                FROM accounts_list
                WHERE
                  LOWER(account_name) LIKE %s
                  OR LOWER(IFNULL(gst_no,'')) LIKE %s
                ORDER BY account_name
                LIMIT %s
                """,
                (like.lower(), like.lower(), int(limit)),
            )

        return cur.fetchall()

    finally:
        cur.close()
        conn.close()


@app.get("/accounts/by_mobile")
def accounts_by_mobile(mobile: str):
    mobile = (mobile or "").strip()
    print("mobile PARAMS:",mobile)
    if not mobile:
        return {}

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    sql = """
        SELECT
            account_id,
            account_name,
            mobile,
            address_line_1 AS addr1,
            address_line_2 AS addr2,
            city,
            pincode AS pin,
            state,
            state_code AS stateCode,
            gst_no AS gstin,
            email,
            phone
        FROM accounts_list
        WHERE mobile = %s
        LIMIT 1
    """

    # 🔍 DEBUG PRINT
    print("[accounts_by_mobile] SQL:", sql.strip())
    print("[accounts_by_mobile] PARAMS:", (mobile,))

    cur.execute(sql, (mobile,))
    row = cur.fetchone()

    cur.close()
    conn.close()
    return row or {}

@app.get("/accounts/suggest_name")
def accounts_suggest_name(q: str = "", limit: int = 10):
    q = (q or "").strip()
    if len(q) < 2:
        return []

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        """
        SELECT
        account_id,
          account_name AS name,
          mobile AS mobile,
          address_line_1 AS addr1,
          address_line_2 AS addr2,
          city AS city,
          pincode AS pin,
          state AS state,
          state_code AS stateCode,
          gst_no AS gstin,
          email AS email,
          phone AS phone
        FROM accounts_list
        WHERE account_name LIKE %s
        ORDER BY account_name
        LIMIT %s
        """,
        (q + "%", int(limit)),
    )

    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

@app.get("/accounts/by_name")
def accounts_by_name(name: str):
    name = (name or "").strip()
    if not name:
        return {}

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute(
        """
        SELECT
        account_id,
          account_name AS name,
          mobile AS mobile,
          address_line_1 AS addr1,
          address_line_2 AS addr2,
          city AS city,
          pincode AS pin,
          state AS state,
          state_code AS stateCode,
          gst_no AS gstin,
          email AS email,
          phone AS phone
        FROM accounts_list
        WHERE account_name = %s
        LIMIT 1
        """,
        (name,),
    )

    row = cur.fetchone()
    cur.close(); conn.close()
    return row or {}

@app.get("/returns/b2c/suggest")
def returns_b2c_suggest(q: str = "", limit: int = 10):
    print("\n--- returns B2C SUGGEST ---")
    q = (q or "").strip()
    if not q:
        return []

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        like = f"%{q}%"
        cur.execute(
    """
    SELECT
  bill_no        AS invoice_no,
  party_name     AS customer_name,
  account_id     AS account_id,
  DATE(date)     AS invoice_date
FROM sales
WHERE
  (
       return_id IS NULL
    OR return_type = 'PARTIAL'
  )
  AND (
       bill_no LIKE %s
    OR party_name LIKE %s
  )
ORDER BY date DESC
LIMIT %s;


    """,
    (like, like, int(limit)),
)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


@app.get("/returns/b2c/bill_items")
def returns_b2c_bill_items(invoice_no: str):
    if not invoice_no:
        raise HTTPException(status_code=400, detail="invoice_no required")

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        # get sale_id
        cur.execute(
            "SELECT id FROM sales WHERE bill_no=%s LIMIT 1",
            (invoice_no,),
        )
        sale = cur.fetchone()
        if not sale:
            raise HTTPException(status_code=404, detail="Bill not found")

        sale_id = sale["id"]

        cur.execute(
            """
            SELECT
              id            AS sale_item_id,
              item_id,
              item_name,
              qty,
              sale_rate     AS rate,
              gst_percent   AS gst
            FROM sale_items
            WHERE sale_id = %s
            ORDER BY row_no
            """,
            (sale_id,),
        )

        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

@app.post("/returns/b2c/save")
def save_b2c_return(payload: B2CReturnPayload):
    if not payload.items:
        raise HTTPException(status_code=400, detail="No return items")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    try:
        # 1️⃣ Fetch sale header
        cur.execute(
            "SELECT id FROM sales WHERE bill_no=%s LIMIT 1",
            (payload.invoice_no,),
        )
        sale = cur.fetchone()
        if not sale:
            raise HTTPException(status_code=404, detail="Sale not found")

        sale_id = sale["id"]

        # 2️⃣ Create return header
        cur.execute(
            """
            INSERT INTO returns
              (sale_id, bill_no, return_date)
            VALUES (%s,%s,NOW())
            """,
            (sale_id, payload.invoice_no),
        )
        return_id = cur.lastrowid

        total_qty = 0
        total_amt = 0

        # 3️⃣ Process each return item
        for it in payload.items:
            if it.return_qty <= 0:
                continue

            # 🔒 Fetch sale item (lock)
            cur.execute(
                """
                SELECT item_id, item_name, qty, sale_rate, gst_percent
                FROM sale_items
                WHERE id=%s
                FOR UPDATE
                """,
                (it.sale_item_id,),
            )
            si = cur.fetchone()
            if not si:
                raise HTTPException(status_code=404, detail="Sale item not found")

            # ❗ Over-return protection
            if it.return_qty > si["qty"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Return qty exceeds sold qty for {si['item_name']}",
                )

            rate = si["sale_rate"]
            amt = it.return_qty * rate

            # 4️⃣ Insert return line
            cur.execute(
                """
                INSERT INTO return_items
                  (return_id, sale_item_id, item_id, item_name,
                   return_qty, sale_rate, gst_percent, return_amount)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    return_id,
                    it.sale_item_id,
                    si["item_id"],
                    si["item_name"],
                    it.return_qty,
                    rate,
                    si["gst_percent"],
                    amt,
                ),
            )

            total_qty += it.return_qty
            total_amt += amt

            # 5️⃣ FIFO reverse
            cur.execute(
                """
                SELECT id, purchase_id, qty_sold
                FROM sale_lines
                WHERE sale_item_id=%s
                ORDER BY id DESC
                FOR UPDATE
                """,
                (it.sale_item_id,),
            )
            fifo_lines = cur.fetchall()

            remaining = it.return_qty
            for ln in fifo_lines:
                if remaining <= 0:
                    break

                used = min(ln["qty_sold"], remaining)

                if ln["purchase_id"]:
                    cur.execute(
                        """
                        UPDATE purchases_s
                        SET remaining_qty = remaining_qty + %s
                        WHERE id=%s
                        """,
                        (used, ln["purchase_id"]),
                    )

                cur.execute(
                    """
                    UPDATE sale_lines
                    SET qty_sold = qty_sold - %s
                    WHERE id=%s
                    """,
                    (used, ln["id"]),
                )

                remaining -= used

            # 6️⃣ Restore item stock
            cur.execute(
                """
                UPDATE items_s
                SET stock_qty = stock_qty + %s
                WHERE id=%s
                """,
                (it.return_qty, si["item_id"]),
            )

        # 7️⃣ Update return totals
        cur.execute(
            """
            UPDATE returns
            SET total_return_qty=%s,
                total_return_amount=%s
            WHERE id=%s
            """,
            (total_qty, total_amt, return_id),
        )

        conn.commit()
        return {
            "ok": True,
            "return_id": return_id,
            "invoice_no": payload.invoice_no,
            "message": "B2C return saved successfully",
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


@app.get("/accounts/by_id")
def accounts_by_id(account_id: int):
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute(
            """
            SELECT
  account_id,
  account_name,
  mobile,
  address_line_1 AS addr1,
  address_line_2 AS addr2,
  city,
  pincode AS pin,
  state,
  state_code AS stateCode,
  gst_no AS gstin,
  email,
  phone
FROM accounts_list
WHERE account_id = %s
LIMIT 1

            """,
            (int(account_id),),
        )

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Account not found")

        return row

    finally:
        cur.close()
        conn.close()


@app.get("/returns/b2c/by_invoice")
def returns_b2c_by_invoice(invoice_no: str):
    if not invoice_no:
        raise HTTPException(status_code=400, detail="invoice_no required")

    print(f"\n🔎 FETCHING B2C RETURN DATA FOR INVOICE: {invoice_no}")

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        # ------------------------------------------------------------
        # 1️⃣ SALE HEADER + ACCOUNT DETAILS
        # ------------------------------------------------------------
        cur.execute(
            """
            SELECT
                s.id,
                s.bill_no            AS invoice_no,
                DATE(s.date)         AS invoice_date,
                s.account_id,
                s.party_name         AS customer_name,

                a.account_name,
                a.mobile,
                a.phone,
                a.gst_no,
                a.address_line_1,
                a.address_line_2,
                a.city,
                a.state,
                a.pincode,
                a.state_code,

                s.taxable,
                s.discount_amount,
                s.gst_amount,
                s.round_off,
                s.grand_total,
                s.paid_now,
                s.balance,
                s.payment_mode
            FROM sales s
            LEFT JOIN accounts_list a
                   ON a.account_id = s.account_id
            WHERE s.bill_no = %s
            LIMIT 1
            """,
            (invoice_no,),
        )

        sale = cur.fetchone()

        print("\n🧾 SALE HEADER RESULT:")
        print(sale)

        if not sale:
            raise HTTPException(status_code=404, detail="Invoice not found")

        sale_id = sale["id"]
        print(f"\n🆔 SALE ID FOUND: {sale_id}")

        # ------------------------------------------------------------
        # 2️⃣ SALE ITEMS + FIFO LINES
        # ------------------------------------------------------------
        cur.execute(
                """
                SELECT
                    si.id                AS sale_item_id,
                    si.item_id,
                    si.row_no,
                    si.item_name,
                    si.unit,
                    si.qty,
                    si.sale_rate         AS rate,
                    si.gst_percent       AS gst,
                    si.line_amount,

                    sl.id                AS sale_line_id,
                    sl.purchase_id,
                    sl.qty_sold,
                    sl.sale_rate         AS fifo_rate
                FROM sale_items si
                LEFT JOIN sale_lines sl
                    ON sl.sale_item_id = si.id
                WHERE si.sale_id = %s
                AND si.return_id IS NULL
                ORDER BY si.row_no, sl.id
                """,
                (sale_id,),
            )


        rows = cur.fetchall()

        print("\n📦 RAW SALE ITEMS + FIFO ROWS:")
        for r in rows:
            print(r)

        # ------------------------------------------------------------
        # 3️⃣ GROUP FIFO UNDER EACH ITEM
        # ------------------------------------------------------------
        items_map = {}

        for r in rows:
            sid = r["sale_item_id"]

            if sid not in items_map:
                items_map[sid] = {
                    "sale_item_id": sid,
                    "item_id": r["item_id"],
                    "item_name": r["item_name"],
                    "unit": r["unit"],
                    "qty": r["qty"],
                    "rate": r["rate"],
                    "gst": r["gst"],
                    "line_amount": r["line_amount"],
                    "fifo_lines": [],
                }

            if r["sale_line_id"]:
                items_map[sid]["fifo_lines"].append({
                    "sale_line_id": r["sale_line_id"],
                    "purchase_id": r["purchase_id"],
                    "qty_sold": r["qty_sold"],
                    "fifo_rate": r["fifo_rate"],
                })

        print("\n🧮 GROUPED ITEMS MAP:")
        for item in items_map.values():
            print(item)

        # ------------------------------------------------------------
        # 4️⃣ FINAL RESPONSE
        # ------------------------------------------------------------
        response = {
            "header": sale,
            "items": list(items_map.values()),
        }

        print("\n✅ FINAL API RESPONSE:")
        print(response)

        return response

    finally:
        cur.close()
        conn.close()
        print("\n🔒 DB CONNECTION CLOSED\n")

class ReturnItem(BaseModel):
    sale_item_id: int
    item_id: Optional[int] = None
    item_name: str
    unit: Optional[str] = None
    qty: float
    rate: float
    gst: float
    line_amount: Optional[float] = None
    fifo_lines: Optional[List[Any]] = []
    return_qty: float

class ReturnHeader(BaseModel):
    id: int
    invoice_no: str
    invoice_date: str
    account_id: int
    customer_name: Optional[str] = None
    mobile: Optional[str] = None
    grand_total: Optional[float] = None


class B2CReturnPayload(BaseModel):
    return_type: str
    header: ReturnHeader
    items: List[ReturnItem]

@app.post("/returns/b2c/createss")
def create_b2c_return(payload: B2CReturnPayload):
    print("\n========== B2C RETURN RECEIVED ==========")

    print("\n--- HEADER ---")
    print(payload.header)

    print("\n--- ITEMS ---")
    for item in payload.items:
        print(item)

    print("\n========================================")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    try:
        # -------------------------------------------------
        # 1️⃣ CALCULATE TOTALS
        # -------------------------------------------------
        total_return_qty = sum(i.return_qty for i in payload.items)
        total_return_amount = sum(i.return_qty * i.rate for i in payload.items)

        # -------------------------------------------------
        # 2️⃣ INSERT INTO returns (MASTER RETURN)
        # -------------------------------------------------
        cur.execute(
            """
            INSERT INTO returns
              (sale_id, bill_no, return_date, reason,
               total_return_qty, total_return_amount)
            VALUES (%s, %s, NOW(), %s, %s, %s)
            """,
            (
                payload.header.id,
                payload.header.invoice_no,
                "B2C Sales Return",
                total_return_qty,
                total_return_amount,
            ),
        )
        return_id = cur.lastrowid

        # -------------------------------------------------
        # 3️⃣ GENERATE RETURN NO
        # -------------------------------------------------
        return_no = f"RET-B2C-{return_id:05d}"

        # -------------------------------------------------
        # 4️⃣ SAFE OPTIONAL HEADER VALUES
        # -------------------------------------------------
        taxable = getattr(payload.header, "taxable", 0.0) or 0.0
        gst_amount = getattr(payload.header, "gst_amount", 0.0) or 0.0
        round_off = getattr(payload.header, "round_off", 0.0) or 0.0

        # -------------------------------------------------
        # 5️⃣ INSERT INTO returns_b2c (B2C HEADER)
        # -------------------------------------------------
        cur.execute(
            """
            INSERT INTO returns_b2c
              (return_no, sale_id, original_bill_no,
               account_id, party_name,
               taxable, gst_amount, total_amount,
               round_off, refund_amount,
               status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                return_no,
                payload.header.id,
                payload.header.invoice_no,
                payload.header.account_id,
                payload.header.customer_name,
                taxable,
                gst_amount,
                total_return_amount,
                round_off,
                total_return_amount,
                "POSTED",
            ),
        )

        # -------------------------------------------------
        # 6️⃣ INSERT INTO returns_b2c_items
        # -------------------------------------------------
        for item in payload.items:
            sold_qty = item.qty
            return_qty = item.return_qty
            rate = item.rate
            gst_percent = item.gst

            taxable_amt = return_qty * rate
            gst_amt = taxable_amt * gst_percent / 100
            total_amt = taxable_amt + gst_amt

            cur.execute(
                """
                INSERT INTO returns_b2c_items
                  (return_id, sale_item_id, item_id, item_name,
                   sold_qty, return_qty,
                   sale_rate, gst_percent,
                   taxable, gst_amount, total_amount, line_amount)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    return_id,
                    item.sale_item_id,
                    item.item_id,
                    item.item_name,
                    sold_qty,
                    return_qty,
                    rate,
                    gst_percent,
                    taxable_amt,
                    gst_amt,
                    total_amt,
                    taxable_amt,
                ),
            )

            return_item_id = cur.lastrowid

            # -------------------------------------------------
            # 7️⃣ INSERT FIFO TRACE (returns_b2c_lines_old)
            # -------------------------------------------------
            remaining = return_qty

            for fl in item.fifo_lines or []:
                if remaining <= 0:
                    break

                used_qty = min(fl["qty_sold"], remaining)

                cur.execute(
                    """
                    INSERT INTO returns_b2c_lines_old
                      (return_item_id,
                       original_sale_item_id,
                       sale_line_id,
                       purchase_id,
                       qty_reversed)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        return_item_id,
                        item.sale_item_id,
                        fl["sale_line_id"],
                        fl["purchase_id"],
                        used_qty,
                    ),
                )

                remaining -= used_qty

                    # -------------------------------------------------
        # 8️⃣ RESTOCK INVENTORY (FIFO REVERSE)
            # -------------------------------------------------

            # 8.1 Restore purchase batch stock
            for fl in item.fifo_lines or []:
                if fl.get("purchase_id"):
                    cur.execute(
                        """
                        UPDATE purchases_s
                        SET remaining_qty = remaining_qty + %s
                        WHERE id = %s
                        """,
                        (fl["qty_sold"], fl["purchase_id"]),
                    )

        # 8.2 Restore item master stock
            cur.execute(
                """
                UPDATE items_s
                SET stock_qty = stock_qty + %s
                WHERE id = %s
                """,
                (item.return_qty, item.item_id),
            )


        # -------------------------------------------------
        # 8️⃣ COMMIT
        # -------------------------------------------------
        conn.commit()

        return {
            "status": "success",
            "return_id": return_id,
            "return_no": return_no,
            "total_return_qty": float(total_return_qty),
            "total_return_amount": float(total_return_amount),
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()

@app.post("/returns/b2c/create")
def create_b2c_return(payload: B2CReturnPayload):
    print("\n========== B2C RETURN RECEIVED ==========")
    print("\n--- HEADER ---")
    print(payload.header)
    print("\n--- ITEMS ---")
    for item in payload.items:
        print(item)
    print("\n========================================")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    try:
        # -------------------------------------------------
        # 1️⃣ CALCULATE TOTALS
        # -------------------------------------------------
        total_return_qty = sum(i.return_qty for i in payload.items)
        total_return_amount = sum(i.return_qty * i.rate for i in payload.items)

        # -------------------------------------------------
        # 2️⃣ INSERT INTO returns (MASTER)
        # -------------------------------------------------
        cur.execute(
            """
            INSERT INTO returns
              (sale_id, bill_no, return_date, reason,
               total_return_qty, total_return_amount)
            VALUES (%s,%s,NOW(),%s,%s,%s)
            """,
            (
                payload.header.id,
                payload.header.invoice_no,
                "B2C Sales Return",
                total_return_qty,
                total_return_amount,
            ),
        )
        return_id = cur.lastrowid

        # -------------------------------------------------
        # 3️⃣ UPDATE sales (INVOICE LEVEL) ✅ FIRST
        # -------------------------------------------------
        cur.execute(
            """
            UPDATE sales
            SET return_id = %s
            WHERE id = %s
            """,
            (
                return_id,
                payload.header.id,
            ),
        )

        # -------------------------------------------------
        # 4️⃣ GENERATE RETURN NO
        # -------------------------------------------------
        return_no = f"RET-B2C-{return_id:05d}"

        # -------------------------------------------------
        # 5️⃣ SAFE HEADER VALUES
        # -------------------------------------------------
        taxable = getattr(payload.header, "taxable", 0.0) or 0.0
        gst_amount = getattr(payload.header, "gst_amount", 0.0) or 0.0
        round_off = getattr(payload.header, "round_off", 0.0) or 0.0

        # -------------------------------------------------
        # 6️⃣ INSERT INTO returns_b2c (HEADER)
        # -------------------------------------------------
        cur.execute(
            """
            INSERT INTO returns_b2c
              (return_no, sale_id, original_bill_no,
               account_id, party_name,
               taxable, gst_amount, total_amount,
               round_off, refund_amount, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                return_no,
                payload.header.id,
                payload.header.invoice_no,
                payload.header.account_id,
                payload.header.customer_name,
                taxable,
                gst_amount,
                total_return_amount,
                round_off,
                total_return_amount,
                "POSTED",
            ),
        )

        # -------------------------------------------------
        # 7️⃣ PROCESS EACH RETURN ITEM
        # -------------------------------------------------
        for item in payload.items:
            sold_qty = item.qty
            return_qty = item.return_qty
            rate = item.rate
            gst_percent = item.gst

            taxable_amt = return_qty * rate
            gst_amt = taxable_amt * gst_percent / 100
            total_amt = taxable_amt + gst_amt

            # 7.1 Insert return item
            cur.execute(
                """
                INSERT INTO returns_b2c_items
                  (return_id, sale_item_id, item_id, item_name,
                   sold_qty, return_qty,
                   sale_rate, gst_percent,
                   taxable, gst_amount, total_amount, line_amount)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    return_id,
                    item.sale_item_id,
                    item.item_id,
                    item.item_name,
                    sold_qty,
                    return_qty,
                    rate,
                    gst_percent,
                    taxable_amt,
                    gst_amt,
                    total_amt,
                    taxable_amt,
                ),
            )
            return_item_id = cur.lastrowid

            # -------------------------------------------------
            # 8️⃣ UPDATE sale_items (ITEM LEVEL) ✅ SECOND
            # -------------------------------------------------
            cur.execute(
                """
                UPDATE sale_items
                SET return_id = %s
                WHERE sale_id = %s
                  AND item_id = %s
                """,
                (
                    return_id,
                    payload.header.id,
                    item.item_id,
                ),
            )

            # -------------------------------------------------
            # 9️⃣ FIFO TRACE + sale_lines UPDATE ✅ THIRD
            # -------------------------------------------------
            remaining = return_qty

            for fl in item.fifo_lines or []:
                if remaining <= 0:
                    break

                used_qty = min(fl["qty_sold"], remaining)

                # FIFO trace
                cur.execute(
                    """
                    INSERT INTO returns_b2c_lines_old
                      (return_item_id,
                       original_sale_item_id,
                       sale_line_id,
                       purchase_id,
                       qty_reversed)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        return_item_id,
                        item.sale_item_id,
                        fl["sale_line_id"],
                        fl["purchase_id"],
                        used_qty,
                    ),
                )

                # sale_lines update
                cur.execute(
                    """
                    UPDATE sale_lines
                    SET return_id = %s
                    WHERE sale_id = %s
                      AND item_id = %s
                      AND purchase_id = %s
                    LIMIT 1
                    """,
                    (
                        return_id,
                        payload.header.id,
                        item.item_id,
                        fl["purchase_id"],
                    ),
                )

                remaining -= used_qty

            # -------------------------------------------------
            # 🔟 RESTOCK INVENTORY
            # -------------------------------------------------
            for fl in item.fifo_lines or []:
                if fl.get("purchase_id"):
                    cur.execute(
                        """
                        UPDATE purchases_s
                        SET remaining_qty = remaining_qty + %s
                        WHERE id = %s
                        """,
                        (fl["qty_sold"], fl["purchase_id"]),
                    )

            cur.execute(
                """
                UPDATE items_s
                SET stock_qty = stock_qty + %s
                WHERE id = %s
                """,
                (return_qty, item.item_id),
            )
            # -------------------------------------------------
             # 🟡 DETERMINE RETURN TYPE (PARTIAL / FULL)
            # -------------------------------------------------

            # Total sold quantity
            cur.execute(
                """
                SELECT COALESCE(SUM(qty), 0) AS total_sold_qty
                FROM sale_items
                WHERE sale_id = %s
                """,
                (payload.header.id,),
            )
            total_sold_qty = cur.fetchone()["total_sold_qty"]

            # ✅ Total returned quantity (ALL returns for this sale)
            cur.execute(
                """
                SELECT COALESCE(SUM(rbi.return_qty), 0) AS total_returned_qty
                FROM returns_b2c_items rbi
                JOIN returns_b2c rb ON rb.id = rbi.return_id
                WHERE rb.sale_id = %s
                """,
                (payload.header.id,),
            )
            total_returned_qty = cur.fetchone()["total_returned_qty"]

            # Decide return type
            return_type = "FULL" if total_returned_qty >= total_sold_qty else "PARTIAL"

            # Update sales
            cur.execute(
                """
                UPDATE sales
                SET return_type = %s
                WHERE id = %s
                """,
                (return_type, payload.header.id),
            )

        # -------------------------------------------------
        # ✅ COMMIT
        # -------------------------------------------------
        conn.commit()

        return {
            "status": "success",
            "return_id": return_id,
            "return_no": return_no,
            "total_return_qty": float(total_return_qty),
            "total_return_amount": float(total_return_amount),
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()


@app.get("/purchases/search")
def purchases_search(q: str = "", limit: int = 50):
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    q = q.strip()
    like = f"%{q}%"

    cur.execute(
        """
        SELECT
          id,
          item_name,
          qty,
          purchase_price,
          selling_price,
          gst_percent,
          bill_no,
          party_name,
          date,
          primary_unit
        FROM purchases_s
        WHERE
          LOWER(item_name) LIKE LOWER(%s)
          OR LOWER(bill_no) LIKE LOWER(%s)
          OR LOWER(party_name) LIKE LOWER(%s)
        ORDER BY date DESC
        LIMIT %s
        """,
        (like, like, like, limit),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows



@app.get("/returns/purchase/suggest")
def purchase_return_suggest(q: str = "", limit: int = 10):
    conn = _get_connection()
    cur = conn.cursor(dictionary=True)
    print("\n--- returns purchasesuggest ---")
    try:
        like = f"%{q}%"
        cur.execute(
            """
            SELECT DISTINCT
                bill_no,
                party_name,
                DATE(date) AS bill_date
            FROM purchases_s
            WHERE bill_no LIKE %s
            ORDER BY date DESC
            LIMIT %s
            """,
            (like, int(limit))
        )
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


@app.get("/returns/purchase/bill")
def purchase_return_by_bill(bill_no: str):
    if not bill_no:
        raise HTTPException(400, "bill_no is required")

    conn = _get_connection()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute(
            """
            SELECT
                id AS purchase_id,
                item_id,
                item_name,
                qty,
                remaining_qty,
                purchase_price,
                gst_percent,
                primary_unit,
                party_name,
                bill_no,
                DATE(date) AS bill_date
            FROM purchases_s
            WHERE bill_no = %s
            ORDER BY id
            """,
            (bill_no,)
        )

        rows = cur.fetchall()
        if not rows:
            raise HTTPException(404, "Purchase bill not found")

        header = {
            "bill_no": rows[0]["bill_no"],
            "party_name": rows[0]["party_name"],
            "date": rows[0]["bill_date"].isoformat()
            if rows[0]["bill_date"] else None,
        }

        items = [
            {
                "purchase_id": r["purchase_id"],
                "item_id": r["item_id"],
                "item_name": r["item_name"],
                "qty": float(r["qty"]),
                "remaining_qty": float(r["remaining_qty"]),
                "purchase_price": float(r["purchase_price"]),
                "gst_percent": float(r["gst_percent"]),
                "primary_unit": r["primary_unit"],
            }
            for r in rows
        ]

        return {"header": header, "items": items}

    finally:
        cur.close()
        conn.close()


@app.post("/returns/purchase")
def return_purchase(p: PurchaseReturnPayload):
    if not p.items:
        raise HTTPException(400, "No return items provided")

    conn = _get_connection()
    conn.autocommit = False
    cur = conn.cursor(dictionary=True)

    try:
        for it in p.items:
            # 1️⃣ Lock purchase batch
            cur.execute("""
                SELECT id, item_id, remaining_qty
                FROM purchases_s
                WHERE id = %s
                FOR UPDATE
            """, (it.purchase_id,))
            pur = cur.fetchone()

            if not pur:
                raise HTTPException(
                    404, f"Purchase row {it.purchase_id} not found"
                )

            if it.return_qty <= 0:
                raise HTTPException(
                    400, f"Invalid return qty for purchase {it.purchase_id}"
                )

            if it.return_qty > pur["remaining_qty"]:
                raise HTTPException(
                    400,
                    f"Return qty {it.return_qty} exceeds available "
                    f"{pur['remaining_qty']} for purchase {it.purchase_id}"
                )

            # 2️⃣ Reduce remaining quantity ONLY
            cur.execute("""
                UPDATE purchases_s
                SET remaining_qty = remaining_qty - %s
                WHERE id = %s
            """, (it.return_qty, it.purchase_id))

            # 3️⃣ Reduce item stock
            cur.execute("""
                UPDATE items_s
                SET stock_qty = stock_qty - %s
                WHERE id = %s
            """, (it.return_qty, pur["item_id"]))

        conn.commit()
        return {"status": "ok"}

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    # -------------------------------
# COMPANY INFO
# -------------------------------

@app.get("/company")
def get_company():
    db = _get_connection()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT * FROM company_info LIMIT 1")
    company = cur.fetchone()

    cur.close()
    db.close()
    return company


@app.post("/company")
def save_company(payload: dict):
    db = _get_connection()
    cur = db.cursor()

    cur.execute("SELECT id FROM company_info LIMIT 1")
    row = cur.fetchone()

    if row:
        cur.execute("""
            UPDATE company_info SET
                company_name=%s,
                gstin=%s,
                state_code=%s,
                address_line1=%s,
                address_line2=%s,
                city=%s,
                state=%s,
                email=%s,
                phone=%s,
                mobile=%s,
                whatsapp=%s
            WHERE id=%s
        """, (
            payload["company_name"],
            payload["gstin"],
            payload["state_code"],
            payload["address_line1"],
            payload["address_line2"],
            payload["city"],
            payload["state"],
            payload["email"],
            payload["phone"],
            payload["mobile"],
            payload["whatsapp"],
            row[0]
        ))
    else:
        cur.execute("""
            INSERT INTO company_info (
                company_name, gstin, state_code,
                address_line1, address_line2,
                city, state,
                email, phone, mobile, whatsapp
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            payload["company_name"],
            payload["gstin"],
            payload["state_code"],
            payload["address_line1"],
            payload["address_line2"],
            payload["city"],
            payload["state"],
            payload["email"],
            payload["phone"],
            payload["mobile"],
            payload["whatsapp"],
        ))

    db.commit()
    cur.close()
    db.close()
    return {"success": True}

# -------------------------------
# COMPANY BANKS
# -------------------------------

@app.get("/company/banks")
def get_banks():
    db = _get_connection()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT * FROM company_banks ORDER BY id")
    rows = cur.fetchall()

    cur.close()
    db.close()
    return rows


@app.post("/company/banks")
def add_bank(payload: dict):
    db = _get_connection()
    cur = db.cursor()

    # Only one default bank allowed
    if payload.get("is_default"):
        cur.execute("UPDATE company_banks SET is_default = FALSE")

    cur.execute("""
        INSERT INTO company_banks
        (company_id, bank_name, branch, ifsc, account_no, is_default)
        VALUES (1,%s,%s,%s,%s,%s)
    """, (
        payload["bank_name"],
        payload["branch"],
        payload["ifsc"],
        payload["account_no"],
        payload.get("is_default", False)
    ))

    db.commit()
    cur.close()
    db.close()
    return {"success": True}


@app.post("/company/banks/{bank_id}/default")
def set_default_bank(bank_id: int):
    db = _get_connection()
    cur = db.cursor()

    cur.execute("UPDATE company_banks SET is_default = FALSE")
    cur.execute(
        "UPDATE company_banks SET is_default = TRUE WHERE id=%s",
        (bank_id,)
    )

    db.commit()
    cur.close()
    db.close()
    return {"success": True}


@app.delete("/company/banks/{bank_id}")
def delete_bank(bank_id: int):
    db = _get_connection()
    cur = db.cursor()

    cur.execute("DELETE FROM company_banks WHERE id=%s", (bank_id,))
    db.commit()

    cur.close()
    db.close()
    return {"success": True}
