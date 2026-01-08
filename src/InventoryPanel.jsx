// src/InventoryPanel.jsx
import React, { useEffect, useRef, useState } from "react";
import {
  apiGetInventory,
  apiSuggestNames,
  apiAddItem,
  apiAddPurchase,
} from "./api";
import "./InventoryPanel.css";

// Format number as Indian currency (for display only)
const toINR = (n) =>
  Number(n || 0).toLocaleString("en-IN", { maximumFractionDigits: 2 });

export default function InventoryPanel() {
  // ==============================
  // 0. TAB STATE
  // ==============================
  const [activeTab, setActiveTab] = useState("purchase"); // "purchase" | "item"

  // ==============================
  // 1. INVENTORY GRID (items table)
  // ==============================
  const [rows, setRows] = useState([]);

  // ==============================
  // 2. ADD ITEM FORM (items table)
  // ==============================
  const [name, setName] = useState("");
  const [groupName, setGroupName] = useState("");
  const [hsnCode, setHsnCode] = useState("");
  const [suggest, setSuggest] = useState([]);
  const debRef = useRef();

  // ==============================
  // 3. PURCHASE CONTEXT
  // ==============================
  const [pDate, setPDate] = useState(() =>
    new Date().toISOString().slice(0, 10)
  );
  const [pBill, setPBill] = useState("");
  const [pParty, setPParty] = useState(""); // Party Name
  const [pGstNo, setPGstNo] = useState(""); // GST Number (GSTIN)
    const [pGst, setPGst] = useState(""); // GST Number (GSTIN)


  // ==============================
  // 4. PURCHASE LINES
  // ==============================
  const makeEmptyLine = () => ({
    item: "",
    qty: "",
    purchase_state: "",
    purchase_price: "",
    purchase_discount: "", // ✅ NEW: discount amount
    gst: pGst || "18", // selling GST
    price: "", // selling price
  });

  const [lines, setLines] = useState([makeEmptyLine()]);
  const [pSuggest, setPSuggest] = useState([]);
  const pDebRef = useRef();

  // ------------------------------
  // Initial load of inventory from DB
  // ------------------------------
  useEffect(() => {
    apiGetInventory()
      .then(setRows)
      .catch(() => {});
  }, []);

  // ------------------------------
  // Keep new line GST in sync with context GST
  // (only updates the trailing empty line's GST)
  // ------------------------------
  useEffect(() => {
    setLines((prev) => {
      if (!prev.length) return [makeEmptyLine()];
      const next = [...prev];
      const lastIdx = next.length - 1;
      const last = next[lastIdx] || {};

      const isLastFilled =
        (last.item || "").trim() !== "" ||
        (last.qty || "") !== "" ||
        (last.purchase_state || "") !== "" ||
        (last.purchase_price || "") !== "" ||
        (last.purchase_discount || "") !== "" ||
        (last.gst || "") !== "" ||
        (last.price || "") !== "";

      if (!isLastFilled) {
        next[lastIdx] = { ...last, gst: pGst || "18" };
      }
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pGst]);

  // ------------------------------
  // Suggestions for "Add Item"
  // ------------------------------
  useEffect(() => {
    clearTimeout(debRef.current);
    debRef.current = setTimeout(async () => {
      if (!name) {
        setSuggest([]);
        return;
      }
      try {
        setSuggest(await apiSuggestNames(name, 25));
      } catch {
        // ignore
      }
    }, 180);

    return () => clearTimeout(debRef.current);
  }, [name]);

  // ------------------------------
  // Suggestions for purchase lines
  // ------------------------------
  const fetchItemSuggestions = (q) => {
    clearTimeout(pDebRef.current);
    pDebRef.current = setTimeout(async () => {
      if (!q) {
        setPSuggest([]);
        return;
      }
      try {
        setPSuggest(await apiSuggestNames(q, 25));
      } catch {
        // ignore
      }
    }, 180);
  };

  // ------------------------------
  // Add new item (with Group + HSN)
  // ------------------------------
  const addItem = async () => {
    const n = (name || "").trim();
    const g = (groupName || "").trim();
    const h = (hsnCode || "").trim();
    if (!n) return alert("Enter item name");
    if (!g) return alert("Enter item group");

    try {
      await apiAddItem(n, h, g);

      const fresh = await apiGetInventory();
      setRows(fresh);

      setName("");
      setGroupName("");
      setHsnCode("");
      setSuggest([]);
    } catch (e) {
      alert("Error while adding item:\n" + e);
    }
  };

  // ------------------------------
  // Update a field within one purchase line
  // ------------------------------
  const updateLine = (idx, key, val) => {
    setLines((prev) => {
      const next = [...prev];
      const row = { ...next[idx] };

      // For numeric-ish fields, allow only digits and dot
      const numericKeys = [
        "qty",
        "purchase_price",
        "purchase_discount",
        "gst",
        "price",
      ];
      if (numericKeys.includes(key)) {
        val = String(val).replace(/[^0-9.]/g, "");
      }

      row[key] = val;
      next[idx] = row;
      return next;
    });

    if (key === "item") fetchItemSuggestions(val);
  };

  // ------------------------------
  // Ensure there is always a trailing empty row
  // ------------------------------
  const ensureTrailingEmptyRow = () => {
    setLines((prev) => {
      const last = prev[prev.length - 1] || {};
      const isLastFilled =
        (last.item || "").trim() !== "" ||
        (last.qty || "") !== "" ||
        (last.purchase_state || "") !== "" ||
        (last.purchase_price || "") !== "" ||
        (last.purchase_discount || "") !== "" ||
        (last.gst || "") !== "" ||
        (last.price || "") !== "";

      return isLastFilled ? [...prev, makeEmptyLine()] : prev;
    });
  };

  // ------------------------------
  // Called after purchase_price is entered (blur/Enter)
  // (kept same behavior as your original)
  // ------------------------------
  const onPurchasePriceDone = (idx) => {
    const r = lines[idx] || {};
    if ((r.item || "").trim() && Number(r.qty) > 0) {
      ensureTrailingEmptyRow();
      setTimeout(() => {
        const el = document.getElementById(`pitem-${idx + 1}`);
        el?.focus();
      }, 0);
    }
  };

  // ------------------------------
  // Remove a purchase line row
  // ------------------------------
  const removeLine = (idx) => {
    setLines((prev) => {
      const next = prev.filter((_, i) => i !== idx);
      return next.length ? next : [makeEmptyLine()];
    });
  };

  // ------------------------------
  // Reset purchase context
  // ------------------------------
  const resetPurchaseContext = () => {
    setPDate(new Date().toISOString().slice(0, 10));
    setPBill("");
    setPParty("");
    setPGstNo("");
  };

  // ------------------------------
  // SUBMIT ALL PURCHASE LINES
  // ------------------------------
  const submitAllPurchases = async () => {
    if (!pBill.trim()) return alert("Enter Purchase Bill No.");

    const payloads = lines
      .map((l) => {
        const item_name = (l.item || "").trim();
        const qty = Number(l.qty);

        // discount amount: stored by reducing purchase_price (no backend change needed)
        const gross = l.purchase_price === "" ? null : Number(l.purchase_price);
        const disc =
          l.purchase_discount === "" ? 0 : Number(l.purchase_discount);

        const netPurchase =
          gross === null ? null : Math.max(0, (gross || 0) - (disc || 0));

        return {
          item_name,
          qty,

          // purchase fields
          purchase_state: (l.purchase_state || "").trim() || null,
          purchase_price: netPurchase,

          // selling fields
          gst_percent: l.gst === "" ? null : Number(l.gst),
          selling_price: l.price === "" ? null : Number(l.price),

          // context fields
          date: pDate,
          bill_no: pBill,
          party_name: (pParty || "").trim() || null,
          gst_no: (pGstNo || "").trim() || null,

        };
      })
      .filter((p) => p.item_name && Number.isFinite(p.qty) && p.qty > 0);

    if (!payloads.length) return alert("Add at least one valid line.");

    try {
      for (const p of payloads) {
        // eslint-disable-next-line no-await-in-loop
        await apiAddPurchase(p);
      }

      const fresh = await apiGetInventory();
      setRows(fresh);

      setLines([makeEmptyLine()]);
    } catch (e) {
      alert("Error while saving purchases:\n" + e);
    }
  };

  // ==============================
  // RENDER
  // ==============================
  return (
    <div className="card">
      {/* === Tabs Header === */}
      <div className="inv-tabs">
        <button
          className={activeTab === "purchase" ? "inv-tab active" : "inv-tab"}
          onClick={() => setActiveTab("purchase")}
        >
          Add Purchase
        </button>
        <button
          className={activeTab === "item" ? "inv-tab active" : "inv-tab"}
          onClick={() => setActiveTab("item")}
        >
          Add Item
        </button>
      </div>

      {/* === TAB: Add Purchase === */}
      {activeTab === "purchase" && (
        <div style={{ marginTop: 12 }}>
          <div className="small" style={{ marginBottom: 6 }}>
            Add Purchase (updates stock & logs in purchases table)
          </div>

          {/* ✅ Context row: Date + Bill No + Party Name + GST + Reset */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr 140px auto",
              gap: 12,
              alignItems: "end",
              marginBottom: 8,
            }}
          >
            <div>
              <label>Date of Purchase</label>
              <input
                type="date"
                value={pDate}
                onChange={(e) => setPDate(e.target.value)}
              />
            </div>

            <div>
              <label>Purchase Bill No.</label>
              <input
                value={pBill}
                onChange={(e) => setPBill(e.target.value)}
                placeholder="PB/1234"
              />
            </div>

            <div>
              <label>Party Name</label>
              <input
                value={pParty}
                onChange={(e) => setPParty(e.target.value)}
                placeholder="Supplier / Party name"
              />
            </div>

            <div>
              <label>GST No</label>
              <input
                value={pGstNo}
                onChange={(e) => setPGstNo(e.target.value.toUpperCase())}
                placeholder="33ABCDE1234F1Z5"
                maxLength={15}
              />
            </div>


            <div className="stack" style={{ alignSelf: "end" }}>
              <button className="btn" onClick={resetPurchaseContext}>
                Reset
              </button>
            </div>
          </div>

          {/* Lines grid */}
          <div style={{ marginTop: 6 }}>
            <table>
              <thead>
                <tr>
                  <th style={{ width: 36 }}>#</th>
                  <th>Item (from items table)</th>
                  <th style={{ width: 110 }}>Quantity</th>
                  <th style={{ width: 130 }}>Purchase State</th>
                  <th style={{ width: 140 }}>Purchase Price</th>
                  <th style={{ width: 120 }}>Discount</th>
                  <th style={{ width: 110 }}>GST %</th>
                  <th style={{ width: 140 }}>Selling Price</th>
                  <th style={{ width: 60 }}></th>
                </tr>
              </thead>

              <tbody>
                {lines.map((r, i) => (
                  <tr key={`line-${i}`}>
                    <td>{i + 1}</td>

                    <td>
                      <input
                        id={`pitem-${i}`}
                        list="inv-suggest-purchase"
                        value={r.item}
                        onChange={(e) => updateLine(i, "item", e.target.value)}
                        placeholder="Start typing to search…"
                        autoComplete="off"
                      />
                    </td>

                    <td>
                      <input
                        value={r.qty}
                        onChange={(e) => updateLine(i, "qty", e.target.value)}
                        placeholder="0"
                      />
                    </td>

                    <td>
                      <input
                        value={r.purchase_state}
                        onChange={(e) =>
                          updateLine(
                            i,
                            "purchase_state",
                            e.target.value.toUpperCase()
                          )
                        }
                        placeholder="00"
                      />
                    </td>

                    <td>
                      <input
                        value={r.purchase_price}
                        onChange={(e) =>
                          updateLine(i, "purchase_price", e.target.value)
                        }
                        placeholder="0.00"
                        onBlur={() => onPurchasePriceDone(i)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") onPurchasePriceDone(i);
                        }}
                      />
                    </td>

                    <td>
                      <input
                        value={r.purchase_discount}
                        onChange={(e) =>
                          updateLine(i, "purchase_discount", e.target.value)
                        }
                        placeholder="0.00"
                      />
                    </td>

                    <td>
                      <input
                        value={r.gst}
                        onChange={(e) => updateLine(i, "gst", e.target.value)}
                        placeholder="18"
                      />
                    </td>

                    <td>
                      <input
                        value={r.price}
                        onChange={(e) => updateLine(i, "price", e.target.value)}
                        placeholder="0.00"
                      />
                    </td>

                    <td className="center">
                      <button
                        className="btn"
                        onClick={() => removeLine(i)}
                        title="Remove row"
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))}

                <tr>
                  <td colSpan={9} className="small">
                    Tip: After entering <strong>Purchase Price</strong>, press{" "}
                    <strong>Enter</strong> to create the next row automatically.
                  </td>
                </tr>
              </tbody>
            </table>

            {/* Shared datalist for purchase items */}
            <datalist id="inv-suggest-purchase">
              {pSuggest.map((s) => (
                <option key={s.name} value={s.name} />
              ))}
            </datalist>

            <div className="stack" style={{ marginTop: 10 }}>
              <button className="btn primary" onClick={submitAllPurchases}>
                Add Purchase
              </button>
            </div>
          </div>
        </div>
      )}

      {/* === TAB: Add Item === */}
      {activeTab === "item" && (
        <div style={{ marginTop: 12 }}>
          <div className="row-3">
            <div>
              <label>Add New Item (or select existing)</label>
              <input
                list="inv-suggest"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., TMT BARS 8MM JSW TMT"
                autoComplete="off"
              />
              <datalist id="inv-suggest">
                {suggest.map((s) => (
                  <option key={s.name} value={s.name} />
                ))}
              </datalist>
            </div>

            <div>
              <label>Group</label>
              <input
                value={groupName}
                onChange={(e) => setGroupName(e.target.value)}
                placeholder="e.g., TMT, MS, CHANNEL, ANGLE"
                autoComplete="off"
              />
            </div>

            <div>
              <label>HSN Code</label>
              <input
                value={hsnCode}
                onChange={(e) => setHsnCode(e.target.value.replace(/[^\d]/g, ""))}
                placeholder="e.g., 7214"
                inputMode="numeric"
                autoComplete="off"
              />
            </div>

            <div className="stack">
              <button className="btn primary" onClick={addItem}>
                Add Item
              </button>
              <button
                className="btn"
                onClick={async () => setRows(await apiGetInventory())}
              >
                Reload from DB
              </button>
            </div>
          </div>
        </div>
      )}

      {/* === Items Grid (current inventory) === */}
      <div style={{ marginTop: 12 }}>
        <table>
          <thead>
            <tr>
              <th style={{ width: 40 }}>#</th>
              <th>Item</th>
              <th className="right">GST %</th>
              <th>Unit</th>
              <th>HSN Code</th>
              <th className="right">Price</th>
              <th className="right">Stock</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.ID || r["ITEM NAME"]}-${i}`}>
                <td>{i + 1}</td>
                <td>{r["ITEM NAME"] || r["ITEM NAM"]}</td>
                <td className="right">{Number(r["GST %"] ?? r["GST"] ?? 0)}</td>
                <td>{r["UNIT"]}</td>
                <td>{r["HSN CODE"] ?? r["HSN"] ?? ""}</td>
                <td className="right">{toINR(r["PRICE"])}</td>
                <td className="right">{toINR(r["STOCK QTY"])}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="center small">
                  No items
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
