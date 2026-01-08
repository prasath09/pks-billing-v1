import React, { useEffect, useState } from "react";
import {
  apiGetCompany,
  apiSaveCompany,
  apiGetCompanyBanks,
  apiAddCompanyBank,
  apiSetDefaultCompanyBank,
  apiDeleteCompanyBank,
} from "./api";

export default function CompanyInfo() {
  const [company, setCompany] = useState({});
  const [banks, setBanks] = useState([]);
  const [bankForm, setBankForm] = useState({
    bank_name: "",
    branch: "",
    ifsc: "",
    account_no: "",
    is_default: false,
  });

  useEffect(() => {
    loadAll();
  }, []);

  async function loadAll() {
    const c = await apiGetCompany();
    const b = await apiGetCompanyBanks();
    setCompany(c || {});
    setBanks(b || []);
  }

  async function saveCompany() {
    await apiSaveCompany(company);
    alert("Company info saved");
  }

  async function addBank() {
    if (!bankForm.bank_name || !bankForm.ifsc || !bankForm.account_no) {
      alert("Bank name, IFSC and Account No are required");
      return;
    }

    await apiAddCompanyBank(bankForm);
    setBankForm({
      bank_name: "",
      branch: "",
      ifsc: "",
      account_no: "",
      is_default: false,
    });
    loadAll();
  }

  async function setDefaultBank(id) {
    await apiSetDefaultCompanyBank(id);
    loadAll();
  }

  async function deleteBank(id) {
    if (!window.confirm("Delete this bank?")) return;
    await apiDeleteCompanyBank(id);
    loadAll();
  }

  return (
    <div className="company-info">
      <h3>Company Info</h3>

      <div className="grid">
        <input
          placeholder="Company Name"
          value={company.company_name || ""}
          onChange={(e) =>
            setCompany({ ...company, company_name: e.target.value })
          }
        />

        <input
          placeholder="GSTIN"
          value={company.gstin || ""}
          onChange={(e) =>
            setCompany({ ...company, gstin: e.target.value })
          }
        />

        <input
          placeholder="State Code"
          value={company.state_code || ""}
          onChange={(e) =>
            setCompany({ ...company, state_code: e.target.value })
          }
        />

        <input
          placeholder="Address Line 1"
          value={company.address_line1 || ""}
          onChange={(e) =>
            setCompany({ ...company, address_line1: e.target.value })
          }
        />

        <input
          placeholder="Address Line 2"
          value={company.address_line2 || ""}
          onChange={(e) =>
            setCompany({ ...company, address_line2: e.target.value })
          }
        />

        <input
          placeholder="City"
          value={company.city || ""}
          onChange={(e) =>
            setCompany({ ...company, city: e.target.value })
          }
        />

        <input
          placeholder="State"
          value={company.state || ""}
          onChange={(e) =>
            setCompany({ ...company, state: e.target.value })
          }
        />

        <input
          placeholder="Email"
          value={company.email || ""}
          onChange={(e) =>
            setCompany({ ...company, email: e.target.value })
          }
        />

        <input
          placeholder="Phone"
          value={company.phone || ""}
          onChange={(e) =>
            setCompany({ ...company, phone: e.target.value })
          }
        />

        <input
          placeholder="Mobile"
          value={company.mobile || ""}
          onChange={(e) =>
            setCompany({ ...company, mobile: e.target.value })
          }
        />

        <input
          placeholder="WhatsApp"
          value={company.whatsapp || ""}
          onChange={(e) =>
            setCompany({ ...company, whatsapp: e.target.value })
          }
        />
      </div>

      <button onClick={saveCompany}>Save Company</button>

      <hr />

      <h4>Bank Details</h4>

      <div className="grid">
        <input
          placeholder="Bank Name"
          value={bankForm.bank_name}
          onChange={(e) =>
            setBankForm({ ...bankForm, bank_name: e.target.value })
          }
        />
        <input
          placeholder="Branch"
          value={bankForm.branch}
          onChange={(e) =>
            setBankForm({ ...bankForm, branch: e.target.value })
          }
        />
        <input
          placeholder="IFSC"
          value={bankForm.ifsc}
          onChange={(e) =>
            setBankForm({ ...bankForm, ifsc: e.target.value })
          }
        />
        <input
          placeholder="Account No"
          value={bankForm.account_no}
          onChange={(e) =>
            setBankForm({ ...bankForm, account_no: e.target.value })
          }
        />
      </div>

      <label>
        <input
          type="checkbox"
          checked={bankForm.is_default}
          onChange={(e) =>
            setBankForm({ ...bankForm, is_default: e.target.checked })
          }
        />
        Set as Default
      </label>

      <br />
      <button onClick={addBank}>Add Bank</button>

      <table className="banks">
        <thead>
          <tr>
            <th>Bank</th>
            <th>Branch</th>
            <th>IFSC</th>
            <th>Account</th>
            <th>Default</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {banks.map((b) => (
            <tr key={b.id}>
              <td>{b.bank_name}</td>
              <td>{b.branch}</td>
              <td>{b.ifsc}</td>
              <td>{b.account_no}</td>
              <td>
                <input
                  type="radio"
                  checked={!!b.is_default}
                  onChange={() => setDefaultBank(b.id)}
                />
              </td>
              <td>
                <button onClick={() => deleteBank(b.id)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
