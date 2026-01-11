// src/CompanyInfoPanel.jsx
import React, { useMemo, useState } from "react";
import "./CompanyInfoPanel.css";

// Fixed company profile (single source of truth)
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
  // In-memory company state only
  const [company, setCompany] = useState(defaultCompany);

  const setCompanyField = (key, value) =>
    setCompany((prev) => ({ ...prev, [key]: value }));

  // -----------------------------
  // Bank form (Add / Edit)
  // -----------------------------
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
    if (!bankForm.bankName || !bankForm.account) {
      alert("Enter bank name and account number");
      return;
    }

    const payload = { ...bankForm, id: bankForm.id ?? Date.now() };

    setCompany((prev) => {
      let banks = [...(prev.banks || [])];

      if (payload.isDefault) {
        banks = banks.map((b) => ({ ...b, isDefault: false }));
      }

      const exists = banks.some((b) => b.id === payload.id);

      banks = exists
        ? banks.map((b) => (b.id === payload.id ? payload : b))
        : [...banks, payload];

      return { ...prev, banks };
    });

    resetBankForm();
  };

  const editBank = (bank) => setBankForm(bank);

  const delBank = (id) =>
    setCompany((prev) => ({
      ...prev,
      banks: prev.banks.filter((b) => b.id !== id),
    }));

  const makeDefault = (id) =>
    setCompany((prev) => ({
      ...prev,
      banks: prev.banks.map((b) => ({
        ...b,
        isDefault: b.id === id,
      })),
    }));

  const defaultBank = useMemo(
    () =>
      company.banks.find((b) => b.isDefault) || company.banks[0],
    [company.banks]
  );

  return (
    <div className="card company-card">
      <div className="small">Company Info (appears on invoice)</div>

      <div className="row-3 mt-6">
        <div>
          <label>Firm / Company Name</label>
          <input value={company.name}
            onChange={(e) => setCompanyField("name", e.target.value)} />
        </div>
        <div>
          <label>GSTIN/UIN</label>
          <input value={company.gstin}
            onChange={(e) => setCompanyField("gstin", e.target.value)} />
        </div>
        <div>
          <label>State Code</label>
          <input value={company.stateCode}
            onChange={(e) => setCompanyField("stateCode", e.target.value)} />
        </div>
      </div>

      <div className="row-3 mt-6">
        <div>
          <label>Address line 1</label>
          <input value={company.addr1}
            onChange={(e) => setCompanyField("addr1", e.target.value)} />
        </div>
        <div>
          <label>Address line 2</label>
          <input value={company.addr2}
            onChange={(e) => setCompanyField("addr2", e.target.value)} />
        </div>
        <div>
          <label>City</label>
          <input value={company.city}
            onChange={(e) => setCompanyField("city", e.target.value)} />
        </div>
      </div>

      <div className="row-3 mt-6">
        <div>
          <label>State</label>
          <input value={company.state}
            onChange={(e) => setCompanyField("state", e.target.value)} />
        </div>
        <div>
          <label>Email</label>
          <input value={company.email}
            onChange={(e) => setCompanyField("email", e.target.value)} />
        </div>
        <div>
          <label>Phone</label>
          <input value={company.phone}
            onChange={(e) => setCompanyField("phone", e.target.value)} />
        </div>
      </div>

      <hr className="sep" />

      <div className="small mb-6">Bank Details</div>

      <table className="company-banks-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Bank</th>
            <th>Branch</th>
            <th>IFSC</th>
            <th>Account</th>
            <th>Default</th>
          </tr>
        </thead>
        <tbody>
          {company.banks.map((b, i) => (
            <tr key={b.id}>
              <td>{i + 1}</td>
              <td>{b.bankName}</td>
              <td>{b.branch}</td>
              <td>{b.ifsc}</td>
              <td>{b.account}</td>
              <td className="center">
                <input
                  type="radio"
                  checked={b.isDefault}
                  onChange={() => makeDefault(b.id)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {defaultBank && (
        <div className="small mt-8">
          Default bank: <b>{defaultBank.bankName}</b> — {defaultBank.account}
        </div>
      )}
    </div>
  );
}
