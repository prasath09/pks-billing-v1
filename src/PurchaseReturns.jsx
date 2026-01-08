import React, { useRef, useState } from "react";
import {
  apiPurchaseReturnSuggest,
  apiPurchaseReturnByBill,
  apiPurchaseReturn,
} from "./api";

export default function PurchaseReturns() {
  // ---------------- HEADER STATE ----------------
  const [billNo, setBillNo] = useState("");
  const [partyName, setPartyName] = useState("");
  const [billDate, setBillDate] = useState("");

  // ---------------- DATA STATE ----------------
  const [items, setItems] = useState([]);
  const [fullBill, setFullBill] = useState(null);

  // ---------------- SUGGEST STATE ----------------
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggest, setShowSuggest] = useState(false);
  const debounceRef = useRef(null);

  // ---------------- FETCH SUGGEST ----------------
  const fetchSuggest = (q) => {
    setBillNo(q);

    if (!q || q.trim().length < 2) {
      setSuggestions([]);
      setShowSuggest(false);
      return;
    }

    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await apiPurchaseReturnSuggest(q, 10);
        setSuggestions(res || []);
        setShowSuggest(true);
      } catch {
        setSuggestions([]);
        setShowSuggest(false);
      }
    }, 300);
  };

  // ---------------- FETCH BILL ----------------
  const fetchBillByNo = async (billNoValue) => {
    if (!billNoValue) {
      alert("Enter purchase bill number");
      return;
    }

    try {
      const res = await apiPurchaseReturnByBill(billNoValue);

      setFullBill(res);
      setPartyName(res.header?.party_name || "");
      setBillDate(res.header?.date ? res.header.date.slice(0, 10) : "");

      setItems(
        (res.items || []).map((i) => ({
          purchase_id: i.purchase_id,
          item_name: i.item_name,
          purchasedQty: Number(i.qty),
          remainingQty: Number(i.remaining_qty),
          rate: Number(i.purchase_price),
          gst: Number(i.gst_percent),
          checked: false,
          returnQty: Number(i.remaining_qty),
        }))
      );
    } catch (err) {
      alert(err.message || "Purchase bill not found");
      setItems([]);
      setFullBill(null);
    }
  };

  // ---------------- SELECT SUGGESTION ----------------
  const selectBill = (row) => {
    setBillNo(row.bill_no);
    setSuggestions([]);
    setShowSuggest(false);
    fetchBillByNo(row.bill_no);
  };

  // ---------------- GRID LOGIC ----------------
  const toggleItem = (id) => {
    setItems((arr) =>
      arr.map((i) =>
        i.purchase_id === id ? { ...i, checked: !i.checked } : i
      )
    );
  };

  const updateQty = (id, val) => {
    setItems((arr) =>
      arr.map((i) =>
        i.purchase_id === id
          ? {
              ...i,
              returnQty: Math.min(Number(val) || 0, i.remainingQty),
            }
          : i
      )
    );
  };

  // ---------------- CANCEL ----------------
  const cancelReturn = () => {
    setItems((arr) =>
      arr.map((i) => ({
        ...i,
        checked: false,
        returnQty: i.remainingQty,
      }))
    );
  };

  // ---------------- SUBMIT RETURN ----------------
  const submitReturn = async () => {
    const selected = items.filter(
      (i) => i.checked && i.returnQty > 0
    );

    if (selected.length === 0) {
      alert("Select at least one item to return");
      return;
    }

    if (!window.confirm("Confirm purchase return?")) return;

    const payload = {
      bill_no: billNo,
      items: selected.map((i) => ({
        purchase_id: i.purchase_id,
        return_qty: i.returnQty,
      })),
    };

    try {
      await apiPurchaseReturn(payload);
      alert("Purchase return completed");
      fetchBillByNo(billNo);
    } catch (err) {
      alert(err.message || "Return failed");
    }
  };

  // ---------------- UI ----------------
  return (
    <>
      <div className="small" style={{ marginBottom: 10 }}>
        📦 <b>Purchase Returns</b>
      </div>

      {/* SEARCH ROW */}
      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          position: "relative",
        }}
      >
        <input
          placeholder="Purchase Bill No"
          value={billNo}
          onChange={(e) => fetchSuggest(e.target.value)}
          style={{ minWidth: 200, height: 34 }}
        />

        <input
          placeholder="Party Name"
          value={partyName}
          readOnly
          style={{ minWidth: 220, height: 34 }}
        />

        <input
          type="date"
          value={billDate}
          readOnly
          style={{ minWidth: 140, height: 34 }}
        />

        <button onClick={() => fetchBillByNo(billNo)}>
          Search Bill
        </button>

        {showSuggest && suggestions.length > 0 && (
          <div style={suggestBox}>
            {suggestions.map((s, i) => (
              <div
                key={i}
                style={suggestItem}
                onMouseDown={() => selectBill(s)}
              >
                <b>{s.bill_no}</b> — {s.party_name}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ITEMS GRID */}
      {items.length > 0 && (
        <>
          <div style={{ marginTop: 20 }}>
            <table width="100%" border="1">
              <thead>
                <tr>
                  <th>Select</th>
                  <th>Item</th>
                  <th>Purchased Qty</th>
                  <th>Available Qty</th>
                  <th>Return Qty</th>
                  <th>Rate</th>
                  <th>Amount</th>
                </tr>
              </thead>
              <tbody>
                {items.map((i) => (
                  <tr key={i.purchase_id}>
                    <td>
                      <input
                        type="checkbox"
                        checked={i.checked}
                        onChange={() => toggleItem(i.purchase_id)}
                      />
                    </td>
                    <td>{i.item_name}</td>
                    <td>{i.purchasedQty}</td>
                    <td>{i.remainingQty}</td>
                    <td>
                      <input
                        type="number"
                        value={i.returnQty}
                        disabled={!i.checked}
                        onChange={(e) =>
                          updateQty(i.purchase_id, e.target.value)
                        }
                        style={{ width: 90 }}
                      />
                    </td>
                    <td>{i.rate}</td>
                    <td>{(i.returnQty * i.rate).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ACTION BUTTONS */}
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              gap: 12,
              marginTop: 16,
            }}
          >
            <button onClick={cancelReturn}>Cancel</button>
            <button
              onClick={submitReturn}
              style={{ background: "#dc2626", color: "#fff" }}
            >
              Return
            </button>
          </div>
        </>
      )}
    </>
  );
}

/* ---------------- STYLES ---------------- */

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
