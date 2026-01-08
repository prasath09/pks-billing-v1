import React, { useRef, useState } from "react";
import {
  apiB2CReturnSuggest,
  apiB2CReturnByInvoice,
  apiB2CReturnCreate,
} from "./api";


export default function B2CReturns() {
   const [invMeta, setInvMeta] = useState({
    date: "",
  });
  // ---------------- BASIC STATE ----------------
  const [invoiceNo, setInvoiceNo] = useState("");
  const [mobile, setMobile] = useState("");
  const [name, setName] = useState("");
  const [invoiceDate, setInvoiceDate] = useState("");
  const [accountId, setAccountId] = useState(null);

  const [billItems, setBillItems] = useState([]);

  // ✅ FULL BACKEND RESPONSE (SOURCE OF TRUTH)
  const [invoiceFull, setInvoiceFull] = useState(null);

  // ---------------- SUGGEST STATE ----------------
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggest, setShowSuggest] = useState(false);
  const debounceRef = useRef(null);

  // ---------------- FETCH BILL ----------------
  const searchBills = async () => {
    if (!invoiceNo) {
      alert("Enter invoice number");
      return;
    }

    try {
      const res = await apiB2CReturnByInvoice(invoiceNo);

      // 🔥 STORE FULL RESPONSE
      setInvoiceFull(res);

      // header (UI)
      setAccountId(res.header?.account_id || null);
      setMobile(res.header?.mobile || "");
      setName(res.header?.customer_name || "");
      setInvoiceDate(
        res.header?.invoice_date
          ? res.header.invoice_date.slice(0, 10)
          : ""
      );

      // grid-friendly UI items
      setBillItems(
        (res.items || []).map((i) => ({
          sale_item_id: i.sale_item_id,
          item_name: i.item_name,
          qty: Number(i.qty),
          rate: Number(i.rate),
          gst: Number(i.gst),
          checked: false,
          returnQty: Number(i.qty),
        }))
      );
    } catch (err) {
      alert(err.message || "Bill not found");
      setBillItems([]);
      setInvoiceFull(null);
    }
  };

  // ---------------- GRID LOGIC ----------------
  const toggleItem = (sale_item_id) => {
    setBillItems((items) =>
      items.map((i) =>
        i.sale_item_id === sale_item_id
          ? { ...i, checked: !i.checked }
          : i
      )
    );
  };

  const updateReturnQty = (sale_item_id, val) => {
    setBillItems((items) =>
      items.map((i) =>
        i.sale_item_id === sale_item_id
          ? {
              ...i,
              returnQty: Math.min(Number(val) || 0, i.qty),
            }
          : i
      )
    );
  };

  // ---------------- SUBMIT RETURN ----------------
  const handleReturn = async () => {
    if (!invoiceFull) {
      alert("Invoice data missing");
      return;
    }

    // UI-selected items
    const selectedUIItems = billItems.filter(
      (i) => i.checked && Number(i.returnQty) > 0
    );

    if (selectedUIItems.length === 0) {
      alert("Select at least one item to return");
      return;
    }

    // 🔥 BUILD ITEMS FROM FULL BACKEND RESPONSE
    const selectedFullItems = invoiceFull.items
      .filter((orig) =>
        selectedUIItems.some(
          (ui) => ui.sale_item_id === orig.sale_item_id
        )
      )
      .map((orig) => {
        const ui = selectedUIItems.find(
          (x) => x.sale_item_id === orig.sale_item_id
        );

        return {
          ...orig,                 // keep fifo_lines, item_id, unit, etc.
          return_qty: ui.returnQty // add return qty
        };
      });

    // ✅ FINAL PAYLOAD (SEND THIS)
    const payload = {
      return_type: "B2C",
      header: {
        ...invoiceFull.header,
      },
      items: selectedFullItems,
    };

    try {
      await apiB2CReturnCreate(payload);
      alert("Return processed successfully");

      // reset
      handleCancel();
    } catch (err) {
      alert(err.message || "Return failed");
    }
  };

  const handleCancel = () => {
    setInvoiceNo("");
    setMobile("");
    setName("");
    setInvoiceDate("");
    setAccountId(null);
    setBillItems([]);
    setInvoiceFull(null);
  };

  // ---------------- SUGGEST LOGIC ----------------
  const fetchSuggestions = (q) => {
    if (!q || q.trim().length < 2) {
      setSuggestions([]);
      setShowSuggest(false);
      return;
    }

    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const rows = await apiB2CReturnSuggest(q, 10);
        setSuggestions(rows || []);
        setShowSuggest(true);
      } catch {
        setSuggestions([]);
        setShowSuggest(false);
      }
    }, 300);
  };

  const selectSuggestion = (row) => {
    setInvoiceNo(row.invoice_no || "");
    setName(row.customer_name || "");
    setInvoiceDate(
      row.invoice_date ? row.invoice_date.slice(0, 10) : ""
    );
    setSuggestions([]);
    setShowSuggest(false);
  };

  // ---------------- UI ----------------
  return (
    <>
      <div className="small" style={{ marginBottom: 8 }}>
        📦 <b>B2C Returns</b> – Customer sales return (without GST invoice)
      </div>

      {/* SEARCH ROW */}
      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          marginTop: 12,
          position: "relative",
        }}
      >
        <input
          style={{ minWidth: 180, height: 34 }}
          placeholder="Invoice Number"
          value={invoiceNo}
          onChange={(e) => {
            setInvoiceNo(e.target.value);
            fetchSuggestions(e.target.value);
          }}
        />

        <input style={{ minWidth: 160, height: 34 }} placeholder="Mobile" value={mobile} readOnly />
        <input style={{ minWidth: 200, height: 34 }} placeholder="Name" value={name} readOnly />
        <input
            type="date"
            value={invMeta.date}
            onChange={(e) =>
              setInvMeta((m) => ({ ...m, date: e.target.value }))
            }
            disabled
            style={{
              height: 34,
              padding: "4px 8px",
              cursor: "pointer",
          }}
        />


        <button onClick={searchBills}>Search Bills</button>

        {showSuggest && suggestions.length > 0 && (
          <div style={suggestBox}>
            {suggestions.map((s, i) => (
              <div
                key={i}
                style={suggestItem}
                onMouseDown={() => selectSuggestion(s)}
              >
                <b>{s.invoice_no}</b> — {s.customer_name}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ITEMS GRID */}
      {billItems.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <table width="100%" border="1">
            <thead>
              <tr>
                <th>Select</th>
                <th>Item</th>
                <th>Sold Qty</th>
                <th>Return Qty</th>
                <th>Rate</th>
                <th>GST %</th>
                <th>Amount</th>
              </tr>
            </thead>
            <tbody>
              {billItems.map((it) => (
                <tr key={it.sale_item_id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={it.checked}
                      onChange={() => toggleItem(it.sale_item_id)}
                    />
                  </td>
                  <td>{it.item_name}</td>
                  <td>{it.qty}</td>
                  <td>
                    <input
                      type="number"
                      value={it.returnQty}
                      disabled={!it.checked}
                      onChange={(e) =>
                        updateReturnQty(it.sale_item_id, e.target.value)
                      }
                    />
                  </td>
                  <td>{it.rate}</td>
                  <td>{it.gst}</td>
                  <td>{(it.returnQty * it.rate).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ textAlign: "right", marginTop: 12 }}>
            <button onClick={handleCancel}>Cancel</button>
            <button onClick={handleReturn} style={{ marginLeft: 10 }}>
              Return
            </button>
          </div>
        </div>
      )}
    </>
  );
}

/* ---------- styles ---------- */
const suggestBox = {
  position: "absolute",
  top: 40,
  left: 0,
  background: "#fff",
  border: "1px solid #d1d5db",
  width: 360,
  zIndex: 50,
};

const suggestItem = {
  padding: "8px 10px",
  cursor: "pointer",
};
