import React, { useEffect, useMemo, useState } from "react";
import BillingPanel from "./BillingPanel";
import AccountsPanel from "./AccountsPanel";
import ItemsPanel from "./ItemsPanel";
import ReturnsPanel from "./ReturnsPanel";


// ===== Styles =====
const styles = `
:root { --border:#d1d5db; --muted:#6b7280; --ink:#111827; --accent:#2563eb; --paper:#fff; }
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; background:#f3f4f6; }
.container { max-width: 1000px; margin: 24px auto; padding: 16px; }
.card { background: var(--paper); border:1px solid var(--border); border-radius: 12px; padding: 16px; }
header { display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px; }
h1 { font-size: 20px; margin:0; }
.muted { color: var(--muted); font-size: 12px; }
.tabs { display:flex; gap:8px; margin: 6px 0 16px; flex-wrap: wrap; }
.tab { padding:8px 12px; border:1px solid var(--border); border-radius:8px; background:#fff; cursor:pointer; font-size:14px; }
.tab.active { border-color: var(--accent); color:white; background: var(--accent); }
label { font-size:12px; color: var(--muted); display:block; margin-bottom:6px; }
input, select, textarea { font-size:14px; width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:8px; background:#fff; }
textarea { resize: vertical; min-height: 76px; }
.row { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
.row-3 { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; }
.row-4 { display:grid; grid-template-columns: 1.3fr .9fr .9fr .9fr; gap:12px; }
.stack { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.btn { padding:10px 14px; border:1px solid var(--border); background:white; border-radius:8px; cursor:pointer; }
.btn.primary { background: var(--accent); color:white; border-color: var(--accent); }
.btn.warn { border-color:#ef4444; color:#ef4444; }
table { width:100%; border-collapse: collapse; }
th, td { border-bottom:1px solid var(--border); padding:8px; text-align:left; vertical-align: top; }
th { font-size:12px; color:var(--muted); font-weight:600; }
.center { text-align:center; }
hr.sep { border:none; border-top:1px solid var(--border); margin: 14px 0; }
.small { font-size:12px; color: var(--muted); }
`;

const defaultCompany = {
  name: "P.KARTHIKEYAN STEEL",
  addr1: "Nethaji Road",
  addr2: "Thiruvarur",
  city: "Thiruvarur",
  state: "Tamil Nadu",
  stateCode: "33",
  gstin: "33AEFSR7230F1ZC",
  email: "karthikeyansteels@gmail.com",
  phone: "",
  mobile: "",
  whatsapp: "",
  banks: [
    {
      id: 1,
      bankName: "TAMILNAD MERCANTILE BANK LTD",
      branch: "SRINIVASAN NAGAR",
      ifsc: "TMBL0000510",
      account: "610701054050002",
      isDefault: true,
    },
  ],
};

export default function App() {
  // ✅ Billing is default active tab
  const [tab, setTab] = useState("billing");

  // Company (defaultStorage)
  const [company, setCompany] = useState(defaultCompany);

  const setCompanyField = (k, v) => setCompany((prev) => ({ ...prev, [k]: v }));

  // Bank form
  const [bankForm, setBankForm] = useState({
    id: null,
    bankName: "",
    branch: "",
    ifsc: "",
    account: "",
    isDefault: false,
  });

  const resetBankForm = () =>
    setBankForm({
      id: null,
      bankName: "",
      branch: "",
      ifsc: "",
      account: "",
      isDefault: false,
    });

  const addOrUpdateBank = () => {
    const payload = { ...bankForm, id: bankForm.id ?? Date.now() };
    if (!payload.bankName || !payload.account) {
      return alert("Enter bank name and account number.");
    }

    setCompany((prev) => {
      let banks = prev.banks ? [...prev.banks] : [];
      const exists = banks.some((b) => b.id === payload.id);

      // If this is default, turn off others
      if (payload.isDefault) banks = banks.map((b) => ({ ...b, isDefault: false }));

      if (exists) banks = banks.map((b) => (b.id === payload.id ? payload : b));
      else banks.push(payload);

      return { ...prev, banks };
    });

    resetBankForm();
  };

  const editBank = (b) => setBankForm(b);

  const delBank = (id) =>
    setCompany((prev) => ({
      ...prev,
      banks: (prev.banks || []).filter((b) => b.id !== id),
    }));

  const makeDefault = (id) =>
    setCompany((prev) => ({
      ...prev,
      banks: (prev.banks || []).map((b) => ({ ...b, isDefault: b.id === id })),
    }));

  const defaultBank = useMemo(() => {
    const banks = company.banks || [];
    return banks.find((b) => b.isDefault) || banks[0] || null;
  }, [company.banks]);

  return (
    <div className="container">
      <style>{styles}</style>

      <div className="card">
        <header>
          <h1>Steel Billing + Inventory</h1>
        </header>

        <div className="tabs">
          <button className={`tab ${tab === "billing" ? "active" : ""}`} onClick={() => setTab("billing")}>
            Billing
          </button>
          <button className={`tab ${tab === "returns" ? "active" : ""}`} onClick={() => setTab("returns")}>
          Returns
          </button>
          <button className={`tab ${tab === "items" ? "active" : ""}`} onClick={() => setTab("items")}>
            Items
          </button>
          <button className={`tab ${tab === "company" ? "active" : ""}`} onClick={() => setTab("company")}>
            Company Info
          </button>
          <button className={`tab ${tab === "accounts" ? "active" : ""}`} onClick={() => setTab("accounts")}>
            Accounts
          </button>
          
        </div>
      </div>

      {tab === "billing" && <BillingPanel company={company} defaultBank={defaultBank} />}

    

      {tab === "company" && (
        <div className="card">
          <div className="small">Company Info (appears on invoice)</div>

          <div className="row-3" style={{ marginTop: 6 }}>
            <div>
              <label>Firm / Company Name</label>
              <input value={company.name} onChange={(e) => setCompanyField("name", e.target.value)} />
            </div>
            <div>
              <label>GSTIN/UIN</label>
              <input value={company.gstin} onChange={(e) => setCompanyField("gstin", e.target.value)} />
            </div>
            <div>
              <label>State Code</label>
              <input value={company.stateCode} onChange={(e) => setCompanyField("stateCode", e.target.value)} />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 6 }}>
            <div>
              <label>Address line 1</label>
              <input value={company.addr1} onChange={(e) => setCompanyField("addr1", e.target.value)} />
            </div>
            <div>
              <label>Address line 2</label>
              <input value={company.addr2} onChange={(e) => setCompanyField("addr2", e.target.value)} />
            </div>
            <div>
              <label>City</label>
              <input value={company.city} onChange={(e) => setCompanyField("city", e.target.value)} />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 6 }}>
            <div>
              <label>State</label>
              <input value={company.state} onChange={(e) => setCompanyField("state", e.target.value)} />
            </div>
            <div>
              <label>Email</label>
              <input value={company.email} onChange={(e) => setCompanyField("email", e.target.value)} />
            </div>
            <div>
              <label>Phone</label>
              <input value={company.phone} onChange={(e) => setCompanyField("phone", e.target.value)} />
            </div>
          </div>

          <div className="row-3" style={{ marginTop: 6 }}>
            <div>
              <label>Mobile</label>
              <input value={company.mobile} onChange={(e) => setCompanyField("mobile", e.target.value)} />
            </div>
            <div>
              <label>WhatsApp</label>
              <input value={company.whatsapp} onChange={(e) => setCompanyField("whatsapp", e.target.value)} />
            </div>
            <div />
          </div>

          <hr className="sep" />
          <div className="small" style={{ marginBottom: 6 }}>
            Bank Details (choose a Default)
          </div>

          <div className="row-4">
            <div>
              <label>Bank Name</label>
              <input value={bankForm.bankName} onChange={(e) => setBankForm((f) => ({ ...f, bankName: e.target.value }))} />
            </div>
            <div>
              <label>Branch</label>
              <input value={bankForm.branch} onChange={(e) => setBankForm((f) => ({ ...f, branch: e.target.value }))} />
            </div>
            <div>
              <label>IFSC</label>
              <input value={bankForm.ifsc} onChange={(e) => setBankForm((f) => ({ ...f, ifsc: e.target.value }))} />
            </div>
            <div>
              <label>Account No</label>
              <input value={bankForm.account} onChange={(e) => setBankForm((f) => ({ ...f, account: e.target.value }))} />
            </div>
          </div>

          <div className="stack" style={{ marginTop: 8 }}>
            <label className="small" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input
                type="checkbox"
                checked={bankForm.isDefault}
                onChange={(e) => setBankForm((f) => ({ ...f, isDefault: e.target.checked }))}
              />
              Set as Default
            </label>

            <button className="btn primary" onClick={addOrUpdateBank}>
              {bankForm.id ? "Update Bank" : "Add Bank"}
            </button>

            {bankForm.id && (
              <button className="btn" onClick={resetBankForm}>
                Cancel
              </button>
            )}
          </div>

          <div style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr>
                  <th style={{ width: 40 }}>#</th>
                  <th>Bank</th>
                  <th>Branch</th>
                  <th>IFSC</th>
                  <th>Account</th>
                  <th className="center">Default</th>
                  <th className="center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(company.banks || []).map((b, i) => (
                  <tr key={b.id}>
                    <td>{i + 1}</td>
                    <td>{b.bankName}</td>
                    <td>{b.branch}</td>
                    <td>{b.ifsc}</td>
                    <td>{b.account}</td>
                    <td className="center">
                      <input type="radio" checked={!!b.isDefault} onChange={() => makeDefault(b.id)} />
                    </td>
                    <td className="center">
                      <button className="btn" onClick={() => editBank(b)}>
                        Edit
                      </button>{" "}
                      <button className="btn warn" onClick={() => delBank(b.id)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
                {(company.banks || []).length === 0 && (
                  <tr>
                    <td colSpan={7} className="center muted">
                      No bank accounts added
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "accounts" && <AccountsPanel />}
      {tab === "items" && <ItemsPanel />}
      {tab === "returns" && <ReturnsPanel />}

    </div>
  );
}
