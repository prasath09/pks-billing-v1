// src/AccountsPanel.jsx
import React, { useEffect, useMemo, useState } from "react";
import { apiAccountsList, apiAccountsUpsert } from "./api";

export default function AccountsPanel() {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  const empty = {
    account_id: "",
    account_name: "",
    group_name: "",
    gst_no: "",
    mobile: "",
    phone: "",
    email: "",
    city: "",
    state: "",
    pincode: "",
    address_line_1: "",
    address_line_2: "",
  };

  const [form, setForm] = useState(empty);

  const load = async () => {
    setLoading(true);
    try {
      const data = await apiAccountsList(q);
      setRows(Array.isArray(data) ? data : []);
    } catch (e) {
      alert("Accounts load failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line
  }, []);

  const onPick = (r) => {
    setForm({
      ...empty,
      ...r,
      account_id: r.account_id ?? "",
      account_name: r.account_name ?? "",
    });
  };

  const setF = (k, v) => setForm((p) => ({ ...p, [k]: v }));

  const save = async () => {
  //  if (!form.account_id) return alert("Account ID required");
    if (!form.account_name) return alert("Account Name required");

    const payload = {
      ...form,
      account_id: Number(form.account_id),
    };

    try {
      await apiAccountsUpsert(payload);
      alert("✅ Saved");
      setForm(empty);
      await load();
    } catch (e) {
      alert("Save failed: " + e.message);
    }
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div className="row-3">
        <div>
          <label>Search Accounts</label>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Type name / GST / mobile..."
          />
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "end" }}>
          <button className="btn" onClick={load} disabled={loading}>
            {loading ? "Loading..." : "Search"}
          </button>
          <button className="btn" onClick={() => { setQ(""); setForm(empty); }}>
            Clear
          </button>
        </div>
        <div />
      </div>

      <hr className="sep" />

      <div className="row-3">
        <div>
          <label>Account ID *</label>
          <input
            value={form.account_id}
            onChange={(e) => setF("account_id", e.target.value)}
            placeholder="e.g. 101"
          />
        </div>
        <div>
          <label>Account Name *</label>
          <input
            value={form.account_name}
            onChange={(e) => setF("account_name", e.target.value)}
            placeholder="Party / Ledger name"
          />
        </div>
        <div>
          <label>Group Name</label>
          <input
            value={form.group_name}
            onChange={(e) => setF("group_name", e.target.value)}
            placeholder="Sundry Debtors / Creditors..."
          />
        </div>
      </div>

      <div className="row-3" style={{ marginTop: 6 }}>
        <div><label>GST No</label><input value={form.gst_no} onChange={(e)=>setF("gst_no", e.target.value)} /></div>
        <div><label>Mobile</label><input value={form.mobile} onChange={(e)=>setF("mobile", e.target.value)} /></div>
        <div><label>Email</label><input value={form.email} onChange={(e)=>setF("email", e.target.value)} /></div>
      </div>

      <div className="row-3" style={{ marginTop: 6 }}>
        <div><label>Address 1</label><input value={form.address_line_1} onChange={(e)=>setF("address_line_1", e.target.value)} /></div>
        <div><label>Address 2</label><input value={form.address_line_2} onChange={(e)=>setF("address_line_2", e.target.value)} /></div>
        <div><label>City</label><input value={form.city} onChange={(e)=>setF("city", e.target.value)} /></div>
      </div>

      <div className="row-3" style={{ marginTop: 6 }}>
        <div><label>State</label><input value={form.state} onChange={(e)=>setF("state", e.target.value)} /></div>
        <div><label>Pincode</label><input value={form.pincode} onChange={(e)=>setF("pincode", e.target.value)} /></div>
        <div style={{ display: "flex", gap: 8, alignItems: "end" }}>
          <button className="btn primary" onClick={save}>Save / Update</button>
        </div>
      </div>

      <hr className="sep" />

      <div className="small" style={{ marginBottom: 8 }}>
        Accounts ({rows.length})
      </div>

      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th style={{ width: 90 }}>ID</th>
              <th>Name</th>
              <th>GST</th>
              <th>Mobile</th>
              <th>City</th>
              <th>State</th>
              <th className="center" style={{ width: 90 }}>Pick</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.account_id}>
                <td>{r.account_id}</td>
                <td>{r.account_name}</td>
                <td>{r.gst_no || ""}</td>
                <td>{r.mobile || ""}</td>
                <td>{r.city || ""}</td>
                <td>{r.state || ""}</td>
                <td className="center">
                  <button className="btn" onClick={() => onPick(r)}>Edit</button>
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={7} className="center muted">No accounts found</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
