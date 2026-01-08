// src/ItemsPanel.jsx
import React, { useEffect, useState } from "react";
import {
  apiItemsDelete,
  apiItemsGetOne,
  apiItemsSearch,
  apiItemsUpsert,
  apiPurchasesSearch,
  apiPurchaseCreate,
  apiPurchasesAdd
} from "./api";

/* ---------------- ITEM MASTER ---------------- */
const emptyItem = () => ({
  id: null,
  item_name: "",
  group_name: "",
  primary_unit: "",
  hsn_code: "",
  tax_percent: "",
  cgst_percent: "",
  sgst_percent: "",
  igst_percent: "",
});

/* ---------------- PURCHASE ---------------- */
const emptyPurchase = () => ({
  item_id: null,
  item_name: "",
  
  qty: "",
  purchase_price: "",
  selling_price: "",
  mrp: "",
  gst_percent: "",
  bill_no: "",
  party_name: "",
  date: "",
});

export default function ItemsPanel() {
  const [activeTab, setActiveTab] = useState("item"); // item | purchase
  const [q, setQ] = useState("");
  const [purchaseQ, setPurchaseQ] = useState("");

  const [rows, setRows] = useState([]);
  const [purchaseRows, setPurchaseRows] = useState([]);

  const [form, setForm] = useState(emptyItem());
  const [purchaseForm, setPurchaseForm] = useState(emptyPurchase());

  const [loading, setLoading] = useState(false);

  // ---------- ITEM SUGGEST (ADD ONLY) ----------
const [itemSuggest, setItemSuggest] = useState([]);
const [showItemSuggest, setShowItemSuggest] = useState(false);

  /* ---------------- LOAD ITEMS ---------------- */
  const loadItems = async (query = "") => {
    setLoading(true);
    try {
      const data = await apiItemsSearch(query);
      setRows(Array.isArray(data) ? data : []);
    } finally {
      setLoading(false);
    }
  };

  /* ---------------- LOAD PURCHASES ---------------- */
  const loadPurchases = async (query = "") => {
    setLoading(true);
    try {
      const data = await apiPurchasesSearch(query);
      setPurchaseRows(Array.isArray(data) ? data : []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadItems("");
  }, []);

  useEffect(() => {
    if (activeTab === "purchase") {
      loadPurchases("");
    }
  }, [activeTab]);

  // ---------- ITEM SUGGEST SEARCH (ADD ONLY) ----------
const searchItemSuggest = async (q) => {
  if (!q) {
    setItemSuggest([]);
    return;
  }
  const data = await apiItemsSearch(q);
  setItemSuggest(Array.isArray(data) ? data.slice(0, 8) : []);
};

  /* ---------------- EDIT ITEM ---------------- */
  const editItem = async (id) => {
    const full = await apiItemsGetOne(id);
    setForm({
      id: full.id,
      item_name: full.item_name || "",
      group_name: full.group_name || "",
      primary_unit: full.primary_unit || "",
      hsn_code: full.hsn_code || "",
      tax_percent: full.tax_percent ?? "",
      cgst_percent: full.cgst_percent ?? "",
      sgst_percent: full.sgst_percent ?? "",
      igst_percent: full.igst_percent ?? "",
    });
    setActiveTab("item");
  };

  /* ---------------- SAVE ITEM ---------------- */
  const saveItem = async () => {
    if (!form.item_name.trim()) {
      alert("Item name is required");
      return;
    }

    await apiItemsUpsert({
      id: form.id,
      item_name: form.item_name.trim(),
      group_name: form.group_name || null,
      primary_unit: form.primary_unit || null,
      hsn_code: form.hsn_code || null,
      tax_percent: form.tax_percent === "" ? null : Number(form.tax_percent),
      cgst_percent: form.cgst_percent === "" ? null : Number(form.cgst_percent),
      sgst_percent: form.sgst_percent === "" ? null : Number(form.sgst_percent),
      igst_percent: form.igst_percent === "" ? null : Number(form.igst_percent),
    });

    alert("Item saved");
    setForm(emptyItem());
    loadItems(q);
  };

  /* ---------------- DELETE ITEM ---------------- */
  const delItem = async (id) => {
    if (!window.confirm("Delete this item?")) return;
    await apiItemsDelete(id);
    loadItems(q);
  };

  /* ---------------- SAVE PURCHASE (UI ONLY) ---------------- */

  const savePurchase = async () => {
  if (!purchaseForm.item_name || !purchaseForm.qty) {
    alert("Item name and quantity are required");
    return;
  }

  if (!purchaseForm.item_id) {
    alert("Please select item from suggestion list");
    return;
  }

  const payload = {
    item_id: purchaseForm.item_id,
    item_name: purchaseForm.item_name,

    qty: Number(purchaseForm.qty),
    remaining_qty: Number(purchaseForm.qty), // FIFO ready

    purchase_price: Number(purchaseForm.purchase_price || 0),
    selling_price: Number(purchaseForm.selling_price || 0),
    mrp: Number(purchaseForm.mrp || 0),

    gst_percent: Number(purchaseForm.gst_percent || 0),

    bill_no: purchaseForm.bill_no || null,
    party_name: purchaseForm.party_name || null,
    date: purchaseForm.date || null,
  };


  console.log("Saving purchase:", payload);
  try {
    await apiPurchasesAdd(payload);

    alert("Purchase saved successfully");

    setPurchaseForm(emptyPurchase());
    loadPurchases(purchaseQ);

  } catch (e) {
    console.error(e);
    alert("Failed to save purchase");
  }
};




  return (
    <div className="card">
      <div className="small">Items</div>

      {/* ---------- TABS ---------- */}
      <div className="tabs" style={{ marginTop: 10 }}>
        <button
          className={activeTab === "item" ? "tab active" : "tab"}
          onClick={() => setActiveTab("item")}
        >
          Add Item
        </button>
        <button
          className={activeTab === "purchase" ? "tab active" : "tab"}
          onClick={() => setActiveTab("purchase")}
        >
          Add Purchase
        </button>
      </div>

      {/* ================= ADD ITEM ================= */}
      {activeTab === "item" && (
        <>
          {/* ---------- ITEM SEARCH ---------- */}
          <div className="row-3" style={{ marginBottom: 10 }}>
            <div>
              <label>Search Items</label>
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Item / HSN / Group"
              />
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "end" }}>
              <button className="btn" onClick={() => loadItems(q)}>
                Search
              </button>
              <button
                className="btn"
                onClick={() => {
                  setQ("");
                  loadItems("");
                }}
              >
                Clear
              </button>
            </div>

            <div />
          </div>

          {/* ---------- ADD / EDIT ITEM FORM ---------- */}
          <div className="row-3">
            <div>
              <label>Item Name *</label>
              <input
                value={form.item_name}
                onChange={(e) =>
                  setForm((f) => ({ ...f, item_name: e.target.value }))
                }
              />
            </div>

            <div>
              <label>Group</label>
              <input
                value={form.group_name}
                onChange={(e) =>
                  setForm((f) => ({ ...f, group_name: e.target.value }))
                }
              />
            </div>

            <div>
              <label>Primary Unit</label>
              <input
                value={form.primary_unit}
                onChange={(e) =>
                  setForm((f) => ({ ...f, primary_unit: e.target.value }))
                }
              />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 8 }}>
            <div>
              <label>HSN Code</label>
              <input
                value={form.hsn_code}
                onChange={(e) =>
                  setForm((f) => ({ ...f, hsn_code: e.target.value }))
                }
              />
            </div>

            <div>
              <label>Tax %</label>
              <input
                value={form.tax_percent}
                onChange={(e) =>
                  setForm((f) => ({ ...f, tax_percent: e.target.value }))
                }
              />
            </div>

            <div>
              <label>IGST %</label>
              <input
                value={form.igst_percent}
                onChange={(e) =>
                  setForm((f) => ({ ...f, igst_percent: e.target.value }))
                }
              />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 8 }}>
            <div>
              <label>CGST %</label>
              <input
                value={form.cgst_percent}
                onChange={(e) =>
                  setForm((f) => ({ ...f, cgst_percent: e.target.value }))
                }
              />
            </div>

            <div>
              <label>SGST %</label>
              <input
                value={form.sgst_percent}
                onChange={(e) =>
                  setForm((f) => ({ ...f, sgst_percent: e.target.value }))
                }
              />
            </div>
            <div />
          </div>

          <div className="stack" style={{ marginTop: 12 }}>
            <button className="btn primary" onClick={saveItem}>
              {form.id ? "Update" : "Save"}
            </button>
          </div>

          <hr className="sep" />

          {/* ---------- ITEM LIST ---------- */}
          <div className="muted">
            Items ({rows.length}) {loading ? "Loading..." : ""}
          </div>

          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Group</th>
                <th>HSN</th>
                <th>Unit</th>
                <th>Tax%</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{r.item_name}</td>
                  <td>{r.group_name}</td>
                  <td>{r.hsn_code}</td>
                  <td>{r.primary_unit}</td>
                  <td>{r.tax_percent}</td>
                  <td>
                    <button className="btn" onClick={() => editItem(r.id)}>
                      Edit
                    </button>{" "}
                    <button
                      className="btn warn"
                      onClick={() => delItem(r.id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="center muted">
                    No items found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </>
      )}

      {/* ================= ADD PURCHASE ================= */}
      {activeTab === "purchase" && (
        <>
          {/* ---------- PURCHASE SEARCH ---------- */}
          <div className="row-3" style={{ marginBottom: 10 }}>
            <div>
              <label>Search Purchases</label>
              <input
                value={purchaseQ}
                onChange={(e) => setPurchaseQ(e.target.value)}
                placeholder="Item / Bill No / Party"
              />
            </div>

            <div style={{ display: "flex", gap: 8, alignItems: "end" }}>
              <button
                className="btn"
                onClick={() => loadPurchases(purchaseQ)}
              >
                Search
              </button>
              <button
                className="btn"
                onClick={() => {
                  setPurchaseQ("");
                  loadPurchases("");
                }}
              >
                Clear
              </button>
            </div>
            <div />
          </div>

          {/* ---------- ADD PURCHASE FORM ---------- */}
          <div className="row-3">
            <div>
             <label>Item Name *</label>
<div style={{ position: "relative" }}>
  <input
    value={purchaseForm.item_name}
    placeholder="Search item..."
    onChange={(e) => {
      const v = e.target.value;
      setPurchaseForm((p) => ({
        ...p,
        item_name: v,
        item_id: null,
      }));
      searchItemSuggest(v);
      setShowItemSuggest(true);
    }}
    onBlur={() => setTimeout(() => setShowItemSuggest(false), 200)}
  />

  {showItemSuggest && itemSuggest.length > 0 && (
    <div className="suggest-box">
      {itemSuggest.map((it) => (
        <div
          key={it.id}
          className="suggest-item"
          onClick={() => {
            setPurchaseForm((p) => ({
              ...p,
              item_id: it.id,
              item_name: it.item_name,
              gst_percent: it.tax_percent ?? "",
            }));
            setShowItemSuggest(false);
          }}
        >
          <b>{it.item_name}</b>
          <div className="muted small">
            ID: {it.id} | HSN: {it.hsn_code}
          </div>
        </div>
      ))}
    </div>
  )}
</div>

            </div>

            <div>
              <label>Quantity *</label>
              <input
                value={purchaseForm.qty}
                onChange={(e) =>
                  setPurchaseForm((p) => ({ ...p, qty: e.target.value }))
                }
              />
            </div>

            <div>
              <label>Purchase Price</label>
              <input
                value={purchaseForm.purchase_price}
                onChange={(e) =>
                  setPurchaseForm((p) => ({
                    ...p,
                    purchase_price: e.target.value,
                  }))
                }
              />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 8 }}>
            <div>
              <label>Selling Price</label>
              <input
                value={purchaseForm.selling_price}
                onChange={(e) =>
                  setPurchaseForm((p) => ({
                    ...p,
                    selling_price: e.target.value,
                  }))
                }
              />
            </div>

            <div>
              <label>MRP</label>
              <input
                value={purchaseForm.mrp}
                onChange={(e) =>
                  setPurchaseForm((p) => ({ ...p, mrp: e.target.value }))
                }
              />
            </div>

            <div>
              <label>GST %</label>
              <input
                value={purchaseForm.gst_percent}
                onChange={(e) =>
                  setPurchaseForm((p) => ({
                    ...p,
                    gst_percent: e.target.value,
                  }))
                }
              />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 8 }}>
            <div>
              <label>Bill No</label>
              <input
                value={purchaseForm.bill_no}
                onChange={(e) =>
                  setPurchaseForm((p) => ({
                    ...p,
                    bill_no: e.target.value,
                  }))
                }
              />
            </div>

            <div>
              <label>Party Name</label>
              <input
                value={purchaseForm.party_name}
                onChange={(e) =>
                  setPurchaseForm((p) => ({
                    ...p,
                    party_name: e.target.value,
                  }))
                }
              />
            </div>

            <div>
              <label>Date</label>
              <input
                type="date"
                value={purchaseForm.date}
                onChange={(e) =>
                  setPurchaseForm((p) => ({ ...p, date: e.target.value }))
                }
              />
            </div>
          </div>

          <div className="stack" style={{ marginTop: 12 }}>
            <button className="btn primary" onClick={savePurchase}>
              Save Purchase
            </button>
          </div>

          <hr className="sep" />

          {/* ---------- PURCHASE LIST ---------- */}
          <div className="muted">
            Purchases ({purchaseRows.length}){" "}
            {loading ? "Loading..." : ""}
          </div>

          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Item</th>
                <th>Qty</th>
                <th>Purchase ₹</th>
                <th>Selling ₹</th>
                <th>GST %</th>
                <th>Bill No</th>
                <th>Party</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {purchaseRows.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.item_name}</td>
                  <td>{p.qty}</td>
                  <td>{p.purchase_price}</td>
                  <td>{p.selling_price}</td>
                  <td>{p.gst_percent}</td>
                  <td>{p.bill_no}</td>
                  <td>{p.party_name}</td>
                  <td>{p.date?.slice(0, 10)}</td>
                </tr>
              ))}

              {purchaseRows.length === 0 && (
                <tr>
                  <td colSpan={9} className="center muted">
                    No purchases found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
