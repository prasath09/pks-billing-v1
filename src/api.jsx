// src/api.js
//const API_BASE = process.env.REACT_APP_API_BASE || "https://billing-pks-production.up.railway.app";
//const API_BASE = "http://localhost:5002";
const API_BASE = "https://billing-pks-production.up.railway.app";
// ---------- helpers ----------
export const toNum = (v) => (isNaN(+v) ? 0 : +v);
export const amt = (q, r) => toNum(q) * toNum(r);
export const inr = (v) =>
  toNum(v).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function normalizeAccountRow(a) {
  if (!a) return a;

  return {
    // identity
    account_id: a.account_id ?? a.id ?? a.accountId ?? null,

    // naming
    name: a.account_name ?? a.name ?? "",

    // contact
    mobile: a.mobile ?? "",
    email: a.email ?? "",
    phone: a.phone ?? "",

    // tax
    gstin: a.gst_no ?? a.gstin ?? "",

    // address
    addr1: a.address1 ?? a.addr1 ?? "",
    addr2: a.address2 ?? a.addr2 ?? "",
    city: a.city ?? "",
    pin: a.pin ? String(a.pin) : "",

    // state
    state: a.state ?? "",
    stateCode: a.state_code ? String(a.state_code) : "",
  };
}


// ================== Items / Inventory (MySQL) ==================
export async function apiGetInventory() {
  return getJSON(`${API_BASE}/inventory`);
}

// Add item in MySQL items table
export async function apiAddItem(name, hsnCode, groupName) {
  const res = await fetch(`${API_BASE}/inventory/add_item`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      hsn_code: hsnCode,
      group: groupName,
    }),
  });
  if (!res.ok) throw new Error("Failed to add item");
  return await res.json();
}


// Suggestions from items table
export async function apiSuggestNames(q = "", limit = 20) {
  const url = new URL(`${API_BASE}/purchases/suggest`);
  if (q) url.searchParams.set("q", q);
  url.searchParams.set("limit", String(limit));
  return getJSON(url.toString()); // [{name,gst,hsn,unit,price}]
}

// Availability check from MySQL
export async function apiCheckAvailability(name, qty = 0) {
  const url = new URL(`${API_BASE}/inventory/availability`);
  url.searchParams.set("name", name || "");
  url.searchParams.set("qty", String(qty || 0));
  const res = await getJSON(`${API_BASE}/inventory/availability?name=${encodeURIComponent(name)}&qty=${qty}`);
  console.log("availability response:", res);
  return getJSON(url.toString()); // {exists,stock,can_deliver}
}

// ================== Purchases (MySQL) ==================
/*
export async function apiAddPurchase({
  item_name,
  qty,
  gst,
  price,
  date,
  bill_no,
}) {
  return postJSON(`${API_BASE}/inventory/add_purchase`, {
    item_name,
    qty: Number(qty || 0),
    gst: Number(gst || 0),
    price: Number(price || 0),
    date,
    bill_no,
  });
}*/
/*
export async function apiAddPurchase({
  item_name,
  qty,
  purchase_state,
  purchase_price,
  gst_percent,     // selling GST
  selling_price,   // selling price
  date,
  bill_no,
}) {
  return postJSON(`${API_BASE}/inventory/add_purchase`, {
    item_name,
    qty: Number(qty || 0),

    // ✅ new fields
    purchase_state: (purchase_state || "").trim() || null,
    purchase_price:
      purchase_price === "" || purchase_price == null
        ? null
        : Number(purchase_price),

    // ✅ selling fields
    gst_percent:
      gst_percent === "" || gst_percent == null ? null : Number(gst_percent),
    selling_price:
      selling_price === "" || selling_price == null ? null : Number(selling_price),

    date,
    bill_no,
  });
}
*/
export async function apiAddPurchase({
  item_name,
  qty,
  purchase_state,
  purchase_price,
  gst_percent,     // selling GST
  selling_price,   // selling price
  date,
  bill_no,
  party_name,      // ✅ NEW
}) {
  return postJSON(`${API_BASE}/inventory/add_purchase`, {
    item_name,
    qty: Number(qty || 0),

    // ✅ NEW fields
    purchase_state: (purchase_state || "").trim() || null,
    purchase_price:
      purchase_price === "" || purchase_price == null
        ? null
        : Number(purchase_price),

    // ✅ selling fields
    gst_percent:
      gst_percent === "" || gst_percent == null ? null : Number(gst_percent),
    selling_price:
      selling_price === "" || selling_price == null
        ? null
        : Number(selling_price),

    // ✅ purchase context
    date,
    bill_no,
    party_name: (party_name || "").trim() || null,  // ✅ NEW
  });
}


// Log a sale row into purchases as negative quantity
export async function apiAddSale(p) {
  return apiAddPurchase({
    item_name: p.item_name,
    qty: -Math.abs(Number(p.qty || 0)),
    gst: Number(p.gst || 0),
    price: Number(p.price || 0),
    date: p.date,
    bill_no: p.bill_no,
  });
}

// Push many billing lines into purchases with negative qty.
export async function apiAddSaleLines(lines = [], meta = {}) {
  const { date, bill_no } = meta;
  const norm = (l) => {
    const item_name = l.itemName ?? l["ITEM NAME"] ?? l.name ?? "";
    const qty = toNum(l.qty ?? l.QTY ?? 0);
    const gst = toNum(l.gst ?? l.gstRate ?? l["GST %"] ?? 0);
    const price = toNum(l.unitPrice ?? l.price ?? l["PRICE"] ?? 0);
    return { item_name, qty, gst, price, date, bill_no };
  };
  for (const line of lines || []) {
    const row = norm(line);
    if (!row.item_name || !row.qty) continue;
    await apiAddSale(row);
  }
}

// Purchases-based name suggest (for billing)
/*export async function apiBillingSuggest(q, limit = 25) {
  const url = new URL(`${API_BASE}/inventory/suggest`);
  url.searchParams.set("q", q || "");
  url.searchParams.set("limit", String(limit));
  return getJSON(url.toString()); // [{name,hsn,gst,price}]
}*/

// Purchases-based name suggest (for billing)
export async function apiBillingSuggest(q) {
  console.log("apiBillingSuggest", q);
  const url = new URL(`${API_BASE}/purchases/suggest`);
  url.searchParams.set("q", q || "");
 // url.searchParams.set("limit", String(limit));
  return getJSON(url.toString()); 
}

export async function apiPurchasesLast(name) {
 console.log("apiPurchasesLast", name);
  const url = new URL(`${API_BASE}/purchases/last`);
  url.searchParams.set("name", name || "");
  return getJSON(url.toString()); // {name,hsn,gst,price,stockQty}
}

export async function apiCreateSaleFIFO(payload) {
  const url = `${API_BASE}/sales/create_fifo`;
  const body = JSON.stringify(payload);

  // ✅ Logs (Developer Console)
  console.group("🟦 apiCreateSaleFIFO → POST /sales/create_fifo");
  console.log("URL:", url);
  console.log("Payload (object):", payload);
  console.log("Payload (JSON pretty):\n", JSON.stringify(payload, null, 2));
  console.log("Body length:", body.length);
  console.groupEnd();

  // ✅ Optional: pause execution to inspect in console
  // debugger;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  // ✅ Also log server response (helpful for debugging 400)
  const text = await res.text();
  console.log("⬅️ Response status:", res.status);
  console.log("⬅️ Response body:", text);

  if (!res.ok) {
    let err = {};
    try {
      err = JSON.parse(text);
    } catch {}
    throw new Error(err.detail || text || "Failed to create sale");
  }

  // parse success JSON
  try {
    return JSON.parse(text);
  } catch {
    return { ok: true, raw: text };
  }
}

// ================== Accounts (MySQL) ==================
export async function apiAccountsList(q = "", limit = 100) {
  const url = new URL(`${API_BASE}/accounts`);
  if (q) url.searchParams.set("q", q);
  url.searchParams.set("limit", String(limit));
  return getJSON(url.toString());
}

export async function apiAccountsUpsert(payload) {
  return postJSON(`${API_BASE}/accounts/upsert`, payload);
}

export async function apiItemsSearch(q = "") {
  const qs = q ? `?q=${encodeURIComponent(q)}` : "";
  return getJSON(`${API_BASE}/items_pk_steel${qs}`);
}

export async function apiItemsGetOne(id) {
  return getJSON(`${API_BASE}/items_pk_steel/${id}`);
}

export async function apiItemsUpsert(payload) {
  return postJSON(`${API_BASE}/items_pk_steel/upsert`, payload);
}

export async function apiItemsDelete(id) {
  const r = await fetch(`${API_BASE}/items_pk_steel/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ================== Stocks ==================
export async function apiGetStocks({ q = "", limit = 200, offset = 0 } = {}) {
  const qs = new URLSearchParams({ q, limit: String(limit), offset: String(offset) });
  return getJSON(`${API_BASE}/stocks?${qs.toString()}`);
}

export async function apiSuggestStock(q) {
  const qs = new URLSearchParams({ q: q || "", limit: "12" });
  return getJSON(`${API_BASE}/stocks/suggest?${qs.toString()}`);
}

export async function apiImportStockExcel(file) {
  const fd = new FormData();
  fd.append("file", file);

  const r = await fetch(`${API_BASE}/stocks/import_excel`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function apiAccountsSuggest(q, limit = 10) {
  const r = await fetch(
    `${API_BASE}/accounts/suggest?q=${encodeURIComponent(q)}&limit=${limit}`
  );
  if (!r.ok) throw new Error(await r.text());
  const rows = await r.json();
  return (rows || []).map(normalizeAccountRow);
}


export async function apiAccountByMobile(mobile) {
  console.log("apiAccountByMobile", mobile);
  const r = await fetch(
    `${API_BASE}/accounts/by_mobile?mobile=${encodeURIComponent(mobile)}`
  );
  if (!r.ok) throw new Error(await r.text());
  const row = await r.json();
  console.log("apiAccountByMobile result:", row);
  return normalizeAccountRow(row);
}

export async function apiAccountsSuggestName(q, limit = 10) {
  const r = await fetch(
    `${API_BASE}/accounts/suggest_name?q=${encodeURIComponent(q)}&limit=${limit}`
  );
  if (!r.ok) throw new Error(await r.text());
  const rows = await r.json();
  return (rows || []).map(normalizeAccountRow);
}

export async function apiAccountByName(name) {
  const r = await fetch(
    `${API_BASE}/accounts/by_name?name=${encodeURIComponent(name)}`
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}



/*
export async function apiCreateSaleFIFO(payload) {
  const res = await fetch(`${API_BASE}/sales/create_fifo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to create sale");
  }

  return res.json();
}
*/

export async function apiAccountById(account_id) {
  const r = await fetch(
    `${API_BASE}/accounts/by_id?account_id=${account_id}`
  );
  if (!r.ok) throw new Error(await r.text());
  return normalizeAccountRow(await r.json());
}

/// ================== Returns (B2C) ==================

export async function apiB2CReturnSuggest(q, limit = 10) {
  const url = new URL(`${API_BASE}/returns/b2c/suggest`);
  url.searchParams.set("q", q || "");
  url.searchParams.set("limit", String(limit));
  return getJSON(url.toString());
}

export async function apiB2CReturnByInvoice(invoice_no) {
  const url = new URL(`${API_BASE}/returns/b2c/by_invoice`);
  url.searchParams.set("invoice_no", invoice_no);
  return getJSON(url.toString());
}

export async function apiB2CReturnCreate(payload) {
  const r = await fetch(`${API_BASE}/returns/b2c/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!r.ok) {
    throw new Error(await r.text());
  }

  return r.json();
}

export async function apiPurchasesSearch(q = "") {
  const url = new URL(`${API_BASE}/purchases/search`);
  url.searchParams.set("q", q);
  return getJSON(url.toString());
}

// src/api.js
export async function apiPurchasesAdd(payload) {
  console.log("add_purchase payload", payload);

  const r = await fetch(`${API_BASE}/inventory/add_purchase`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!r.ok) {
    throw new Error(await r.text());
  
  }

  return r.json();
}

// ================== Purchase Returns ==================

// ================== Purchase Returns ==================

// 🔍 Suggest purchase bills
export async function apiPurchaseReturnSuggest(q = "", limit = 10) {
  const url = new URL(`${API_BASE}/returns/purchase/suggest`);
  url.searchParams.set("q", q || "");
  url.searchParams.set("limit", String(limit));
  return getJSON(url.toString());
}

// 📄 Fetch full purchase bill
export async function apiPurchaseReturnByBill(billNo) {
  const url = new URL(`${API_BASE}/returns/purchase/bill`);
  url.searchParams.set("bill_no", billNo);
  return getJSON(url.toString());
}

// 🔁 Submit purchase return (ONE request, ALL items)
export async function apiPurchaseReturn(payload) {
  return postJSON(`${API_BASE}/returns/purchase`, payload);
}

// ================== Company Info ==================

export async function apiGetCompany() {
  return getJSON(`${API_BASE}/company`);
}

export async function apiSaveCompany(payload) {
  return postJSON(`${API_BASE}/company`, payload);
}

export async function apiGetCompanyBanks() {
  return getJSON(`${API_BASE}/company/banks`);
}

export async function apiAddCompanyBank(payload) {
  return postJSON(`${API_BASE}/company/banks`, payload);
}

export async function apiSetDefaultCompanyBank(bankId) {
  return postJSON(`${API_BASE}/company/banks/${bankId}/default`, {});
}

export async function apiDeleteCompanyBank(bankId) {
  const r = await fetch(`${API_BASE}/company/banks/${bankId}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
