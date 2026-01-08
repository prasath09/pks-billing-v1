// src/CompanyInfoPanel.jsx
import React, { useEffect, useMemo, useState } from "react";
import "./CompanyInfoPanel.css"; // ⬅️ new CSS file

// Default company profile used on first load (or if localStorage is empty)
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

export default function CompanyInfoPanel() {
  // -----------------------------
  // Company profile (localStorage)
  // -----------------------------
  const [company, setCompany] = useState(() => {
    // On first render, try to load from localStorage
    const ls = localStorage.getItem("company.v1");
    return ls ? JSON.parse(ls) : defaultCompany;
  });

  // Whenever `company` changes, persist to localStorage
//  useEffect(() => {
  //  localStorage.setItem("company.v1", JSON.stringify(company));
  //}, [company]);

useEffect(() => {
  const ls = localStorage.getItem("company.v1");
  if (!ls) {
    localStorage.setItem("company.v1", JSON.stringify(defaultCompany));
  }
}, []);


  // Helper to update a single field in the `company` object
  const setCompanyField = (key, value) =>
    setCompany((prev) => ({ ...prev, [key]: value }));

  // -----------------------------
  // Bank form state (Add / Edit)
  // -----------------------------
  const [bankForm, setBankForm] = useState({
    id: null,          // null = new bank; non-null = editing existing bank
    bankName: "",
    branch: "",
    ifsc: "",
    account: "",
    isDefault: false,
  });

  // Reset bank form to empty (used after save / cancel)
  const resetBankForm = () =>
    setBankForm({
      id: null,
      bankName: "",
      branch: "",
      ifsc: "",
      account: "",
      isDefault: false,
    });

  // Add a new bank OR update existing bank
  const addOrUpdateBank = () => {
    // If editing existing bank, keep its id; if new, use Date.now()
    const payload = { ...bankForm, id: bankForm.id ?? Date.now() };

    // Basic validation
    if (!payload.bankName || !payload.account)
      return alert("Enter bank name and account number.");

    setCompany((prev) => {
      let banks = prev.banks ? [...prev.banks] : [];
      const exists = banks.some((b) => b.id === payload.id);

      // If this bank is set as default, unset default on all others
      if (payload.isDefault) {
        banks = banks.map((b) => ({ ...b, isDefault: false }));
      }

      // Update existing bank or push new bank
      if (exists) {
        banks = banks.map((b) => (b.id === payload.id ? payload : b));
      } else {
        banks.push(payload);
      }

      return { ...prev, banks };
    });

    // Clear the form after save
    resetBankForm();
  };

  // Load a bank into the form for editing
  const editBank = (bank) => setBankForm(bank);

  // Delete a bank by id
  const delBank = (id) =>
    setCompany((prev) => ({
      ...prev,
      banks: (prev.banks || []).filter((b) => b.id !== id),
    }));

  // Mark a given bank as default (only one default at a time)
  const makeDefault = (id) =>
    setCompany((prev) => ({
      ...prev,
      banks: (prev.banks || []).map((b) => ({
        ...b,
        isDefault: b.id === id,
      })),
    }));

  // Compute the default bank (memoized)
  const defaultBank = useMemo(
    () =>
      (company.banks || []).find((b) => b.isDefault) ||
      (company.banks || [])[0],
    [company.banks]
  );

  return (
    <div className="card company-card">
      {/* Header note */}
      <div className="small">Company Info (appears on invoice)</div>

      {/* Row 1: Name, GSTIN, State code */}
      <div className="row-3 mt-6">
        <div>
          <label>Firm / Company Name</label>
          <input
            value={company.name}
            onChange={(e) => setCompanyField("name", e.target.value)}
          />
        </div>
        <div>
          <label>GSTIN/UIN</label>
          <input
            value={company.gstin}
            onChange={(e) => setCompanyField("gstin", e.target.value)}
          />
        </div>
        <div>
          <label>State Code</label>
          <input
            value={company.stateCode}
            onChange={(e) => setCompanyField("stateCode", e.target.value)}
          />
        </div>
      </div>

      {/* Row 2: Address lines + City */}
      <div className="row-3 mt-6">
        <div>
          <label>Address line 1</label>
          <input
            value={company.addr1}
            onChange={(e) => setCompanyField("addr1", e.target.value)}
          />
        </div>
        <div>
          <label>Address line 2</label>
          <input
            value={company.addr2}
            onChange={(e) => setCompanyField("addr2", e.target.value)}
          />
        </div>
        <div>
          <label>City</label>
          <input
            value={company.city}
            onChange={(e) => setCompanyField("city", e.target.value)}
          />
        </div>
      </div>

      {/* Row 3: State, Email, Phone */}
      <div className="row-3 mt-6">
        <div>
          <label>State</label>
          <input
            value={company.state}
            onChange={(e) => setCompanyField("state", e.target.value)}
          />
        </div>
        <div>
          <label>Email</label>
          <input
            value={company.email}
            onChange={(e) => setCompanyField("email", e.target.value)}
          />
        </div>
        <div>
          <label>Phone</label>
          <input
            value={company.phone}
            onChange={(e) => setCompanyField("phone", e.target.value)}
          />
        </div>
      </div>

      {/* Row 4: Mobile, WhatsApp */}
      <div className="row-3 mt-6">
        <div>
          <label>Mobile</label>
          <input
            value={company.mobile}
            onChange={(e) => setCompanyField("mobile", e.target.value)}
          />
        </div>
        <div>
          <label>WhatsApp</label>
          <input
            value={company.whatsapp}
            onChange={(e) => setCompanyField("whatsapp", e.target.value)}
          />
        </div>
        <div>{/* empty third column */}</div>
      </div>

      {/* Separator line */}
      <hr className="sep" />

      {/* Bank section heading */}
      <div className="small mb-6">
        Bank Details (choose a Default)
      </div>

      {/* Bank form (Add / Edit) */}
      <div className="row-4">
        <div>
          <label>Bank Name</label>
          <input
            value={bankForm.bankName}
            onChange={(e) =>
              setBankForm((f) => ({ ...f, bankName: e.target.value }))
            }
          />
        </div>
        <div>
          <label>Branch</label>
          <input
            value={bankForm.branch}
            onChange={(e) =>
              setBankForm((f) => ({ ...f, branch: e.target.value }))
            }
          />
        </div>
        <div>
          <label>IFSC</label>
          <input
            value={bankForm.ifsc}
            onChange={(e) =>
              setBankForm((f) => ({ ...f, ifsc: e.target.value }))
            }
          />
        </div>
        <div>
          <label>Account No</label>
          <input
            value={bankForm.account}
            onChange={(e) =>
              setBankForm((f) => ({ ...f, account: e.target.value }))
            }
          />
        </div>
      </div>

      {/* Bank form actions */}
      <div className="stack mt-8">
        <label className="small checkbox-inline">
          <input
            type="checkbox"
            checked={bankForm.isDefault}
            onChange={(e) =>
              setBankForm((f) => ({ ...f, isDefault: e.target.checked }))
            }
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

      {/* Bank list table */}
      <div className="mt-12">
        <table className="company-banks-table">
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
                  <input
                    type="radio"
                    checked={!!b.isDefault}
                    onChange={() => makeDefault(b.id)}
                  />
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
                <td colSpan={7} className="center small">
                  No bank accounts added
                </td>
              </tr>
            )}
          </tbody>
        </table>

        {defaultBank && (
          <div className="small mt-8">
            Default bank: <b>{defaultBank.bankName}</b> — {defaultBank.account}
          </div>
        )}
      </div>
    </div>
  );
}
