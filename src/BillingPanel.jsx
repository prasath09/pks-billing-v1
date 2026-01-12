// src/BillingPanel.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import * as XLSX from "xlsx";
import "./BillingPanel.css";

import {
  apiBillingSuggest,
  apiCheckAvailability,
  apiPurchasesLast,
  apiCreateSaleFIFO,
  apiAccountsSuggest,
  apiAccountByMobile,
  apiAccountsSuggestName,
  apiAccountById,
} from "./api";


// ---- helpers (scoped to panel) ----
const emptyLine = () => ({
  itemName: "",
  qty: "",
  unitPrice: "",
  hsn: "",
  gst: "",
  unit: "KG",
});

const fmtDDMMYYYY = (iso) => {
  const d = new Date(iso);
  const dd = String(d.getDate()).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const yy = d.getFullYear();
  return `${dd}-${mm}-${yy}`;
};

function toINR(n) {
  const v = isNaN(n) ? 0 : n;
  return v.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function numberToWordsIndian(num) {
  const a = [
    "",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
    "Fourteen",
    "Fifteen",
    "Sixteen",
    "Seventeen",
    "Eighteen",
    "Nineteen",
  ];
  const b = [
    "",
    "",
    "Twenty",
    "Thirty",
    "Forty",
    "Fifty",
    "Sixty",
    "Seventy",
    "Eighty",
    "Ninety",
  ];
  function inWords(n) {
    if (n < 20) return a[n];
    if (n < 100)
      return b[Math.floor(n / 10)] + (n % 10 ? " " + a[n % 10] : "");
    if (n < 1000)
      return (
        a[Math.floor(n / 100)] +
        " Hundred" +
        (n % 100 ? " and " + inWords(n % 100) : "")
      );
    return "";
  }
  if (num === 0) return "Zero";
  const parts = [];
  const units = [
    ["Crore", 10000000],
    ["Lakh", 100000],
    ["Thousand", 1000],
    ["", 1],
  ];
  let n = Math.floor(num);
  for (const [label, value] of units) {
    const q = Math.floor(n / value);
    if (q) {
      if (value >= 1000) {
        if (q < 100) parts.push(inWords(q) + " " + label);
        else parts.push(numberToWordsIndian(q) + " " + label);
      } else {
        parts.push(inWords(q));
      }
      n = n % value;
    }
  }
  return parts.join(" ");
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const normCode = (v) => String(v || "").trim();
const isSameState = (companyStateCode, customerStateCode) =>
  normCode(companyStateCode) !== "" &&
  normCode(customerStateCode) !== "" &&
  normCode(companyStateCode) === normCode(customerStateCode);

// small numeric helper
const n = (v) => (isNaN(Number(v)) || !v ? 0 : Number(v));
// Mobile-based typeahead (accounts_list)


/** Billing panel */
export default function BillingPanel({ company, defaultBank, inventory }) {
  // Buyer / consignee / meta
// Payment receipt reference
const [paymentReceiptNo, setPaymentReceiptNo] = useState("");


  const [buyer, setBuyer] = useState({
      account_id: null,
    name: "",
    mobile: "",  
    addr1: "",
    addr2: "",
    city: "",
    pin: "",
    state: "",
    stateCode: "",
    gstin: "",
    email: "",
    phone: "",
  });

  // Selected account (ledger)
const [selectedAccount, setSelectedAccount] = useState(null);
const [invoiceType, setInvoiceType] = useState("B2C"); // default

// Account search
const [accQuery, setAccQuery] = useState("");
const [accSuggest, setAccSuggest] = useState([]);
const accDebRef = useRef();
const normalizeAccount = (a) => ({
  ...a,
   account_id: a.account_id ?? a.id ?? null, 
  name: a.name ?? a.account_name ?? "",
});

useEffect(() => {
  // 🚫 HARD STOP: ledger already selected
  if (selectedAccount) {
    setAccSuggest([]);
    return;
  }

  clearTimeout(accDebRef.current);
  accDebRef.current = setTimeout(async () => {
    const q = (accQuery || "").trim();
    if (q.length < 2) {
      setAccSuggest([]);
      return;
    }

    try {
      const rows = await apiAccountsSuggestName(q, 10);
      setAccSuggest((rows || []).map(normalizeAccount));
    } catch {
      setAccSuggest([]);
    }
  }, 180);

  return () => clearTimeout(accDebRef.current);
}, [accQuery, selectedAccount]);


/*

useEffect(() => {
  clearTimeout(accDebRef.current);
  accDebRef.current = setTimeout(async () => {
    const q = (accQuery || "").trim();
    if (q.length < 2) return setAccSuggest([]);
    try {
      const rows = await apiAccountsSuggestName(q, 10);
      setAccSuggest((rows || []).map(normalizeAccount));
    } catch {
      setAccSuggest([]);
    }
  }, 180);

  return () => clearTimeout(accDebRef.current);
}, [accQuery]);

*/
  // Payments
  const [advance, setAdvance] = useState("0"); // kept in state, but field hidden now
  const [paymentMode, setPaymentMode] = useState("");
  const paymentModes = [
    "Select",
    "Cash",
    "UPI",
    "Card",
    "Bank Transfer",
    "Cheque",
    "Credit",
    "Online",
  ];

  // Split payments
  const [splitEnabled, setSplitEnabled] = useState(false);
  const [split1, setSplit1] = useState({ amount: "", mode: "Select", ref: "" });
  const [split2, setSplit2] = useState({ amount: "", mode: "Select", ref: "" });

  const [consigneeDifferent, setConsigneeDifferent] = useState(false);
  const [consignee, setConsignee] = useState({ ...buyer });
  useEffect(() => {
    if (!consigneeDifferent) setConsignee({ ...buyer });
  }, [consigneeDifferent, buyer]);

  const todayIso = new Date().toISOString().slice(0, 10);
  const [invMeta, setInvMeta] = useState({
    invoiceNo: "PKS-B-00001",
    date: todayIso,
    ewayNo: "",
    vehicleNo: "",
    kms: "",
    fromDate: "",
    toDate: "",
    lrDate: todayIso,
});


  // lines + discount
  const [lines, setLines] = useState([emptyLine()]);
  const [discountPct, setDiscountPct] = useState("0");

  // Purchases-based type-ahead
  const [suggest, setSuggest] = useState([]); // [{name}]
  const [activeQuery, setActiveQuery] = useState("");
  const debRef = useRef();

  useEffect(() => {
    clearTimeout(debRef.current);
    debRef.current = setTimeout(async () => {
      const q = (activeQuery || "").trim();
      if (!q) return setSuggest([]);
      try {
        const list = await apiBillingSuggest(q, 25);
        setSuggest(list || []);
      } catch {
        setSuggest([]);
      }
    }, 180);
    return () => clearTimeout(debRef.current);
  }, [activeQuery]);

  const [mobileSuggest, setMobileSuggest] = useState([]); // rows from backend
const [mobileQuery, setMobileQuery] = useState("");
const mobDebRef = useRef();

  useEffect(() => {
  clearTimeout(mobDebRef.current);
  mobDebRef.current = setTimeout(async () => {
    const q = (mobileQuery || "").trim();
    if (q.length < 2) return setMobileSuggest([]);
    try {
      const rows = await apiAccountsSuggest(q, 10);
      setMobileSuggest((rows || []).map(normalizeAccount));
    } catch {
      setMobileSuggest([]);
    }
  }, 180);

  return () => clearTimeout(mobDebRef.current);
}, [mobileQuery]);

const [nameSuggest, setNameSuggest] = useState([]);
const [nameQuery, setNameQuery] = useState("");
const nameDebRef = useRef();

useEffect(() => {
  clearTimeout(nameDebRef.current);
  nameDebRef.current = setTimeout(async () => {
    const q = (nameQuery || "").trim();
    if (q.length < 2) return setNameSuggest([]);
    try {
      const rows = await apiAccountsSuggestName(q, 10);
      setNameSuggest((rows || []).map(normalizeAccount));
    } catch {
      setNameSuggest([]);
    }
  }, 180);

  return () => clearTimeout(nameDebRef.current);
}, [nameQuery]);

useEffect(() => {
  if (invoiceType === "B2C") {
    setBuyer((b) => ({ ...b, gstin: "" }));
  }
}, [invoiceType]);

<input
  value={buyer.gstin}
  disabled={invoiceType === "B2C"}
  onChange={(e) =>
    setBuyer((b) => ({ ...b, gstin: e.target.value }))
  }
/>


const pickName = async (acc) => {
  if (!acc) return;

  setNameQuery(acc.name || "");
  setNameSuggest([]);

  try {
    // 🔑 Resolve FULL ledger by ID (or mobile fallback)
    let full = null;

    if (acc.account_id || acc.id) {
      full = await apiAccountById(acc.account_id ?? acc.id);
    } else if (acc.mobile) {
      full = await apiAccountByMobile(acc.mobile);
    }

    if (!full) return;

    const normalized = {
      ...full,
      account_id: full.account_id ?? full.id ?? null,
      name: full.name ?? full.account_name ?? "",
    };

    // ✅ SINGLE SOURCE OF TRUTH
    setSelectedAccount(normalized);
    setAccQuery(normalized.name);
    fillBuyerFromAccount(normalized);
  } catch (e) {
    console.error("pickName failed:", e);
  }
};

const pickAccount = async (acc) => {
  console.log("Clicked account:", acc);

  if (!acc?.account_id) return;

  try {
    const full = await apiAccountById(acc.account_id);

    if (!full) return;

    const normalized = {
      ...full,
      name: full.name ?? full.account_name ?? "",
    };

    setSelectedAccount(normalized);
    setAccQuery(normalized.name);
    fillBuyerFromAccount(normalized);
  
  } catch (e) {
    console.error("Account fetch failed", e);
  } finally {
    setAccSuggest([]);
  }
};

/*
const fillBuyerFromAccount = (acc) => {
  
  if (!acc) return;
   console.log("fillBuyerFromAccount acc:", acc); // 🔍 debug

  setBuyer((b) => ({
    ...b,
    account_id: acc.account_id ?? b.account_id,
     name: acc.account_name ?? acc.name ?? b.name,
    mobile: acc.mobile ?? b.mobile,
    addr1: acc.addr1 ?? b.addr1,
    addr2: acc.addr2 ?? b.addr2,
    city: acc.city ?? b.city,
    pin: String(acc.pin ?? b.pin ?? "").replace(/[^0-9]/g, ""),
    state: acc.state ?? b.state,
    stateCode: String(acc.stateCode ?? b.stateCode ?? "").replace(/[^0-9]/g, ""),
    gstin: acc.gstin ?? b.gstin,
    email: acc.email ?? b.email,
    phone: acc.phone ?? b.phone,
  }));
};
*/
const fillBuyerFromAccount = (acc) => {
  if (!acc) return;

  const normalized = {
    account_id: acc.account_id ?? acc.id ?? null,
    name: acc.account_name ?? acc.name ?? "",
    mobile: acc.mobile ?? "",
  };

  console.log("fillBuyerFromAccount acc:", acc);

  setBuyer((b) => ({
    ...b,
    account_id: normalized.account_id,
    name: normalized.name,
    mobile: acc.mobile ?? b.mobile,
    addr1: acc.addr1 ?? b.addr1,
    addr2: acc.addr2 ?? b.addr2,
    city: acc.city ?? b.city,
    pin: String(acc.pin ?? b.pin ?? "").replace(/[^0-9]/g, ""),
    state: acc.state ?? b.state,
    stateCode: String(acc.stateCode ?? b.stateCode ?? "").replace(/[^0-9]/g, ""),
    gstin: acc.gstin ?? b.gstin,
    email: acc.email ?? b.email,
    phone: acc.phone ?? b.phone,
  }));

  // ✅ THIS IS THE MISSING LINK
  setSelectedAccount(normalized);
  setAccQuery(normalized.name);
};
  // per-line availability status: { exists, stock, can_deliver }
  const [statuses, setStatuses] = useState({});

  const updateAvailability = async (idx, name, qty) => {
    const vname = (name || "").trim();
    const q = Number(qty ?? 0);
    if (!vname) {
      setStatuses((s) => ({ ...s, [idx]: undefined }));
      return;
    }
    try {
      const st = await apiCheckAvailability(vname, Number.isFinite(q) ? q : 0);
      setStatuses((s) => ({ ...s, [idx]: st }));
    } catch {
      setStatuses((s) => ({ ...s, [idx]: undefined }));
    }
  };

  const lookupFromInventory = (val) => {
    const arr = Array.isArray(inventory) ? inventory : [];
    const byLower = (s) => String(s || "").toLowerCase();

    let found = arr.find((i) => byLower(i.name) === byLower(val));
    if (found) {
      return {
        name: found.name,
        hsn: found.hsn ?? found["HSN"] ?? found["HSN CODE"] ?? "",
        gst: Number(found.gst ?? found["GST %"] ?? 0) || 0,
        unit: found.unit ?? found["UNIT"] ?? "KGS",
        price: Number(found.price ?? found["PRICE"] ?? 0) || 0,
      };
    }

    const f2 = arr.find((i) => byLower(i["ITEM NAME"]) === byLower(val));
    if (f2) {
      return {
        name: f2["ITEM NAME"],
        hsn: f2["HSN"] ?? f2["HSN CODE"] ?? "",
        gst: Number(f2["GST %"] ?? 0) || 0,
        unit: f2["UNIT"] ?? "KGS",
        price: Number(f2["PRICE"] ?? 0) || 0,
      };
    }
    return null;
  };

  const handleItemChange = async (idx, val) => {
    setActiveQuery(val);

    let picked = null;
    try {
      if ((val || "").trim()) {
        picked = await apiPurchasesLast(val.trim());
      }
    } catch {
      /* ignore */
    }

    if (!picked) {
      const item = lookupFromInventory(val);
      picked = item
        ? {
            name: item.name,
            hsn: item.hsn || "",
            price: Number(item.price || 0),
            gst: Number(item.gst || 0),
            stockQty: 0,
          }
        : null;
    }

    setLines((prev) => {
      const next = prev.map((r, i) =>
        i !== idx
          ? r
          : {
              ...r,
              itemName: val,
              hsn: picked?.hsn ?? r.hsn,
              unit: "KG",
              unitPrice: String(picked?.price ?? r.unitPrice),
              gst: String(picked?.gst ?? r.gst),
            }
      );
      const isNonEmpty = !!(next[idx].itemName || next[idx].qty);
      const hasBlank =
        next.length &&
        !next[next.length - 1].itemName &&
        !next[next.length - 1].qty;
      if (isNonEmpty && !hasBlank) next.push(emptyLine());
      return next;
    });

    const qtyNow = Number(lines[idx]?.qty || 0);
    updateAvailability(idx, val, qtyNow);
    setSuggest([]);
  };

  const handleQty = (idx, v) => {
    const clean = String(v).replace(/[^0-9.]/g, "");
    setLines((prev) => {
      const next = prev.map((r, i) =>
        i !== idx ? r : { ...r, qty: clean }
      );
      const isNonEmpty = !!(next[idx].itemName || clean);
      const hasBlank =
        next.length &&
        !next[next.length - 1].itemName &&
        !next[next.length - 1].qty;
      if (isNonEmpty && !hasBlank) next.push(emptyLine());
      return next;
    });
    updateAvailability(idx, lines[idx]?.itemName || "", clean);
  };

  const handlePrice = (idx, v) =>
    setLines((prev) =>
      prev.map((r, i) =>
        i !== idx
          ? r
          : { ...r, unitPrice: String(v).replace(/[^0-9.]/g, "") }
      )
    );
  const handleHSN = (idx, v) =>
    setLines((prev) =>
      prev.map((r, i) => (i !== idx ? r : { ...r, hsn: v }))
    );
  const handleGST = (idx, v) =>
    setLines((prev) =>
      prev.map((r, i) =>
        i !== idx
          ? r
          : { ...r, gst: String(v).replace(/[^0-9.]/g, "") }
      )
    );
  const handleUnit = (idx, v) =>
    setLines((prev) =>
      prev.map((r, i) => (i !== idx ? r : { ...r, unit: v }))
    );
  const addRow = () => setLines((prev) => [...prev, emptyLine()]);
  const removeRow = (idx) => {
    setLines((prev) => prev.filter((_, i) => i !== idx));
    setStatuses((s) => {
      const { [idx]: _omit, ...rest } = s;
      return rest;
    });
  };

  // ============ totals (with GST rounding) ============
  const computed = useMemo(() => {
    const filled = lines.filter((l) => l.itemName && Number(l.qty) > 0);

    const baseRows = filled.map((l) => {
      const qty = Number(l.qty || 0);
      const rate = Number(l.unitPrice || 0);
      const gstRate = Number(l.gst || 0);
      const lineTaxable = qty * rate;
      return { ...l, qty, rate, gstRate, lineTaxable };
    });

    const taxable = baseRows.reduce((s, r) => s + r.lineTaxable, 0);

    const pct = Math.max(0, Number(discountPct || 0));
    const discountAmount = +(taxable * (pct / 100)).toFixed(2);
    const discountedTaxable = +(taxable - discountAmount).toFixed(2);

    const factor = taxable > 0 ? 1 - pct / 100 : 1;

    let cgstRaw = 0,
      sgstRaw = 0,
      igstRaw = 0;
    baseRows.forEach((r) => {
      const discLineTaxable = r.lineTaxable * factor;
      const lineGST = discLineTaxable * (r.gstRate / 100);
      if (
        (buyer.stateCode || "").trim() !==
        (company?.stateCode || "").trim()
      ) {
        igstRaw += lineGST;
      } else {
        cgstRaw += lineGST / 2;
        sgstRaw += lineGST / 2;
      }
    });

    const rawGSTTotal = cgstRaw + sgstRaw + igstRaw;

    const roundRupee = (x) => Math.round(x);
    const isInter =
      (buyer.stateCode || "").trim() !==
      (company?.stateCode || "").trim();
    const cgstRounded = isInter ? 0 : roundRupee(cgstRaw);
    const sgstRounded = isInter ? 0 : roundRupee(sgstRaw);
    const igstRounded = isInter ? roundRupee(igstRaw) : 0;
    const gstRounded = isInter ? igstRounded : cgstRounded + sgstRounded;

    const roundOff = +(gstRounded - rawGSTTotal).toFixed(2);
    const total = +(discountedTaxable + gstRounded).toFixed(2);
    const grand = total;

    return {
      rows: baseRows,
      taxable,
      discountAmount,
      discountedTaxable,
      rawGSTTotal,
      cgstRounded,
      sgstRounded,
      igstRounded,
      gstRounded,
      roundOff,
      total,
      grand,
    };
  }, [lines, discountPct, buyer.stateCode, company?.stateCode]);

  const isInterState = useMemo(
    () =>
      (buyer.stateCode || "").trim() !==
      (company?.stateCode || "").trim(),
    [buyer.stateCode, company?.stateCode]
  );

  // Split: paid now & balance
  const paidNow = useMemo(() => {
    const s1 = splitEnabled ? n(split1.amount) : 0;
    const s2 = splitEnabled ? n(split2.amount) : 0;
    return n(advance) + s1 + s2;
  }, [advance, splitEnabled, split1.amount, split2.amount]);

  const balanceAfterSplits = useMemo(
    () => Math.max(0, +(computed.grand - paidNow).toFixed(2)),
    [computed.grand, paidNow]
  );

  // ====== Export Excel (with split details) ======
  const exportExcel = () => {
    const consigneeForExcel = consigneeDifferent ? consignee : buyer;

    const pct = Math.max(0, Number(discountPct || 0));
    const factor = 1 - pct / 100;

    const invoiceAdvance = n(advance);
    const split1Amt = splitEnabled ? n(split1.amount) : 0;
    const split2Amt = splitEnabled ? n(split2.amount) : 0;
    const invoicePaidNow = invoiceAdvance + split1Amt + split2Amt;
    const invoiceBalance = +(computed.grand - invoicePaidNow).toFixed(2);

    const rows = (computed.rows.length ? computed.rows : []).map((r, idx) => {
      const lineTaxableDisc = +(r.lineTaxable * factor).toFixed(2);
      const lineGSTRaw = lineTaxableDisc * (r.gstRate / 100);

      const cgstP = isInterState ? 0 : r.gstRate / 2;
      const sgstP = isInterState ? 0 : r.gstRate / 2;
      const igstP = isInterState ? r.gstRate : 0;

      const totalVal = lineTaxableDisc + lineGSTRaw;
      const roundOffForRow = idx === 0 ? computed.roundOff : 0;

      return {
        "VOUCHER TYPE": "Sales",
        DATE: fmtDDMMYYYY(invMeta.date),
        "VOUCHER NUMBER": invMeta.invoiceNo,
        "PARTY NAME": buyer.name || "",
        GSTIN: buyer.gstin || "",
        ADDRESS: `${buyer.addr1 || ""} ${buyer.addr2 || ""}`.trim(),
        STATE: buyer.state || "",
        "BUYER CITYPIN": buyer.city || "",
        "BUYER STATE CODE": buyer.stateCode || "",
        "CONSIGNEE NAME": consigneeForExcel.name || "",
        "CONSIGNEE ADDRESS": `${consigneeForExcel.addr1 || ""} ${
          consigneeForExcel.addr2 || ""
        }`.trim(),
        "CONSIGNEE CITYPIN": consigneeForExcel.city || "",
        "CONSIGNEE STATE": consigneeForExcel.state || "",
        "CONSIGNEE STATE CODE": consigneeForExcel.stateCode || "",
        "ITEM NAME": r.itemName || "",
        "HSN CODE": r.hsn || "",
        QTY: r.qty || 0,
        UOM: r.unit || "KG",
        RATE: r.rate || 0,
        "TAXABLE VALUE": +lineTaxableDisc.toFixed(2),
        "CGST %": cgstP,
        "SGST %": sgstP,
        "IGST %": igstP,
        "TOTAL VALUE": +totalVal.toFixed(2),
        NARRATION: paymentMode
          ? `Sale of goods — Payment: ${paymentMode}` +
            (splitEnabled
              ? `; Split1 ${split1Amt} ${split1.mode}${
                  split1.ref ? " (" + split1.ref + ")" : ""
                }; Split2 ${split2Amt} ${split2.mode}${
                  split2.ref ? " (" + split2.ref + ")" : ""
                }`
              : "")
          : splitEnabled
          ? `Sale of goods — Split1 ${split1Amt} ${split1.mode}${
              split1.ref ? " (" + split1.ref + ")" : ""
            }; Split2 ${split2Amt} ${split2.mode}${
              split2.ref ? " (" + split2.ref + ")" : ""
            }`
          : "Sale of goods",
        "ROUND OFF": roundOffForRow,
        "PLACE OF SUPPLY": buyer.state || "",
        "TERMS OF PAYMENT": paymentMode || "",
        "EWAY BILL NO": invMeta.ewayNo || "",
        "LR DATE": invMeta.lrDate || "",
        "INV TAXABLE TOTAL": +computed.taxable.toFixed(2),
        "INV DISCOUNT %": pct,
        "INV DISCOUNT AMOUNT": +computed.discountAmount.toFixed(2),
        "INV TAXABLE AFTER DISC": +computed.discountedTaxable.toFixed(2),
        "INV GST TOTAL": +computed.gstRounded.toFixed(2),
        "INV ROUND OFF": +computed.roundOff.toFixed(2),
        "INV GRAND TOTAL": +computed.grand.toFixed(2),
        "INV ADVANCE": invoiceAdvance,
        "INV SPLIT1 AMT": split1Amt,
        "INV SPLIT1 MODE": splitEnabled ? split1.mode : "",
        "INV SPLIT1 REF": splitEnabled ? split1.ref : "",
        "INV SPLIT2 AMT": split2Amt,
        "INV SPLIT2 MODE": splitEnabled ? split2.mode : "",
        "INV SPLIT2 REF": splitEnabled ? split2.ref : "",
        "INV PAID NOW": invoicePaidNow,
        "INV BALANCE": invoiceBalance,
      };
    });

    const data =
      rows.length > 0
        ? rows
        : [
            {
              "VOUCHER TYPE": "Sales",
              DATE: fmtDDMMYYYY(invMeta.date),
              "VOUCHER NUMBER": invMeta.invoiceNo,
              "PARTY NAME": buyer.name || "",
              GSTIN: buyer.gstin || "",
              ADDRESS: `${buyer.addr1 || ""} ${buyer.addr2 || ""}`.trim(),
              STATE: buyer.state || "",
              "BUYER CITYPIN": buyer.city || "",
              "BUYER STATE CODE": buyer.stateCode || "",
              "CONSIGNEE NAME": consigneeForExcel.name || "",
              "CONSIGNEE ADDRESS": `${consigneeForExcel.addr1 || ""} ${
                consigneeForExcel.addr2 || ""
              }`.trim(),
              "CONSIGNEE CITYPIN": consigneeForExcel.city || "",
              "CONSIGNEE STATE": consigneeForExcel.state || "",
              "CONSIGNEE STATE CODE": consigneeForExcel.stateCode || "",
              "ITEM NAME": "",
              "HSN CODE": "",
              QTY: 0,
              UOM: "KGS",
              RATE: 0,
              "TAXABLE VALUE": 0,
              "CGST %": 0,
              "SGST %": 0,
              "IGST %": 0,
              "TOTAL VALUE": 0,
              NARRATION: paymentMode
                ? `Sale of goods — Payment: ${paymentMode}`
                : "Sale of goods",
              "ROUND OFF": computed.roundOff || 0,
              "PLACE OF SUPPLY": buyer.state || "",
              "TERMS OF PAYMENT": paymentMode || "",
              "EWAY BILL NO": invMeta.ewayNo || "",
              "LR DATE": invMeta.lrDate || "",
              "INV TAXABLE TOTAL": +computed.taxable.toFixed(2),
              "INV DISCOUNT %": pct,
              "INV DISCOUNT AMOUNT": +computed.discountAmount.toFixed(2),
              "INV TAXABLE AFTER DISC":
                +computed.discountedTaxable.toFixed(2),
              "INV GST TOTAL": +computed.gstRounded.toFixed(2),
              "INV ROUND OFF": +computed.roundOff.toFixed(2),
              "INV GRAND TOTAL": +computed.grand.toFixed(2),
              "INV ADVANCE": invoiceAdvance,
              "INV SPLIT1 AMT": split1Amt,
              "INV SPLIT1 MODE": splitEnabled ? split1.mode : "",
              "INV SPLIT1 REF": splitEnabled ? split1.ref : "",
              "INV SPLIT2 AMT": split2Amt,
              "INV SPLIT2 MODE": splitEnabled ? split2.mode : "",
              "INV SPLIT2 REF": splitEnabled ? split2.ref : "",
              "INV PAID NOW": invoicePaidNow,
              "INV BALANCE": invoiceBalance,
            },
          ];

    const ws = XLSX.utils.json_to_sheet(data, {
      header: [
        "VOUCHER TYPE",
        "DATE",
        "VOUCHER NUMBER",
        "PARTY NAME",
        "GSTIN",
        "ADDRESS",
        "STATE",
        "BUYER CITYPIN",
        "BUYER STATE CODE",
        "CONSIGNEE NAME",
        "CONSIGNEE ADDRESS",
        "CONSIGNEE CITYPIN",
        "CONSIGNEE STATE",
        "CONSIGNEE STATE CODE",
        "ITEM NAME",
        "HSN CODE",
        "QTY",
        "UOM",
        "RATE",
        "TAXABLE VALUE",
        "CGST %",
        "SGST %",
        "IGST %",
        "TOTAL VALUE",
        "NARRATION",
        "ROUND OFF",
        "PLACE OF SUPPLY",
        "TERMS OF PAYMENT",
        "EWAY BILL NO",
        "LR DATE",
        "INV TAXABLE TOTAL",
        "INV DISCOUNT %",
        "INV DISCOUNT AMOUNT",
        "INV TAXABLE AFTER DISC",
        "INV GST TOTAL",
        "INV ROUND OFF",
        "INV GRAND TOTAL",
        "INV ADVANCE",
        "INV SPLIT1 AMT",
        "INV SPLIT1 MODE",
        "INV SPLIT1 REF",
        "INV SPLIT2 AMT",
        "INV SPLIT2 MODE",
        "INV SPLIT2 REF",
        "INV PAID NOW",
        "INV BALANCE",
      ],
    });

    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Invoice");
    const fileName = `Invoice_${invMeta.invoiceNo.replace(
      /[\/\\:*?"<>|]/g,
      "-"
    )}.xlsx`;
    XLSX.writeFile(wb, fileName);
  };

  // ===================== PRINT (with split payment details) =====================
  const printNow = () => {
    const missing = [];
    if (!buyer.name.trim()) missing.push("Buyer Name");
    if (!buyer.addr1.trim()) missing.push("Buyer Address 1");
    if (!buyer.city.trim()) missing.push("Buyer City");
    if (!buyer.pin?.trim?.()) missing.push("Buyer PIN");

    if (missing.length) {
      alert("Please fill mandatory fields:\n" + missing.join("\n"));
      return;
    }

    const issues = [];
    lines.forEach((l, i) => {
      if (!l.itemName || !Number(l.qty)) return;
      const st = statuses[i];
      if (!st?.exists) issues.push(`Row ${i + 1}: Item not found in inventory`);
      else if (!st?.can_deliver)
        issues.push(`Row ${i + 1}: Only ${st.stock} in stock`);
    });
    if (issues.length) {
      alert("Please fix before printing:\n" + issues.join("\n"));
      return;
    }

    const wnd = window.open("", "_blank");
    if (!wnd) return;

    const useConsignee = consigneeDifferent ? consignee : buyer;
    const fmtL = (d) =>
      new Date(d).toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      });

    // build invoice rows for print
    const rowHtmlArr = (computed.rows.length ? computed.rows : []).map(
      (r, i) => {
        const pct = Math.max(0, Number(discountPct || 0));
        const factor = 1 - pct / 100;

        const lineBase = r.lineTaxable;
        const lineTaxableDisc = +(r.lineTaxable * factor).toFixed(2);
        const amountWithTaxRaw =
          lineTaxableDisc + (lineTaxableDisc * r.gstRate) / 100;

        const taxLabel = r.gstRate
          ? isInterState
            ? `${r.gstRate}% IGST`
            : `${(r.gstRate / 2).toFixed(0)}% + ${(r.gstRate / 2).toFixed(
                0
              )}%`
          : "-";

        return `
        <tr>
          <td class="c">${i + 1}</td>
          <td>${escapeHtml(r.itemName)}</td>
          <td class="c">${escapeHtml(r.hsn || "")}</td>
          <td class="r">${r.qty}</td>
          <td class="c">${escapeHtml(r.unit || "KGS")}</td>
          <td class="r">${toINR(r.rate)}</td>
          <td class="r">${toINR(lineBase)}</td>
          <td class="c">${taxLabel}</td>
          <td class="r">${toINR(amountWithTaxRaw)}</td>
        </tr>`;
      }
    );

    // pad rows to fixed count (10)
    const TARGET_ROWS = 10;
    const blankCount = Math.max(0, TARGET_ROWS - rowHtmlArr.length);
    for (let k = 0; k < blankCount; k++) {
      rowHtmlArr.push(`
        <tr>
          <td class="c">&nbsp;</td>
          <td>&nbsp;</td>
          <td class="c">&nbsp;</td>
          <td class="r">&nbsp;</td>
          <td class="c">&nbsp;</td>
          <td class="r">&nbsp;</td>
          <td class="r">&nbsp;</td>
          <td class="c">&nbsp;</td>
          <td class="r">&nbsp;</td>
        </tr>`);
    }

    const rowsHtml =
      rowHtmlArr.join("") ||
      `<tr><td colspan="9" class="c muted">No items</td></tr>`;

    const cssHref =
      Array.from(document.styleSheets)
        .map((s) => s.href)
        .filter(Boolean)
        .find((h) => h.includes("BillingPanel.css")) || "BillingPanel.css";

    // totals used in print
    const adv = paidNow; // advance + split payments
    const balanceDue = balanceAfterSplits;

    // NEW: split details rows
    const splitLinesHtml = splitEnabled
      ? `
            <tr><td colspan="8" class="r">Split #1 (${escapeHtml(
              split1.mode || "-"
            )})</td><td class="r">${toINR(n(split1.amount))}</td></tr>
            ${
              n(split2.amount) > 0
                ? `<tr><td colspan="8" class="r">Split #2 (${escapeHtml(
                    split2.mode || "-"
                  )})</td><td class="r">${toINR(n(split2.amount))}</td></tr>`
                : ""
            }
        `
      : "";

    const html = `<!doctype html><html><head><meta charset="utf-8" />
      <title>Tax Invoice</title>
      <link rel="stylesheet" href="${cssHref}">
      <style>
        @page { size: A4; margin: 10mm; }
        @media print {
          html, body { width: 210mm; }
          .shell { border: 0; padding: 0; }
          table { page-break-inside: auto; }
          tr { page-break-inside: avoid; break-inside: avoid; }
          thead { display: table-header-group; }
          tfoot { display: table-footer-group; }
        }
        body{font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:16px}
        .small{font-size:12px;color:#555}
        .muted{color:#6b7280}
        .r{text-align:right}.c{text-align:center}
        .shell{border:1px solid #111; padding:12px}
        .grid{display:grid; grid-template-columns: 1.2fr .9fr; gap:12px}
        .box{border:1px solid #111; padding:8px}
        .two{display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px}
        .foot{display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px}
        .hr{border-top:1px solid #ddd; margin:6px 0}
        table{width:100%; border-collapse:collapse; margin-top:8px}
        th,td{border:1px solid #111; padding:6px; font-size:13px}
        th{background:#f9fafb; white-space:nowrap}
        td:nth-child(2){word-break:break-word}
        tbody tr:nth-child(odd){background:#fcfcfc}
      </style>
    </head>
    <body class="invoice-print">
      <div class="shell">
        <div class="grid">
          <div class="box">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <div>
                <div style="font-size:18px;font-weight:700">${escapeHtml(
                  company?.name || ""
                )}</div>
                <div class="small">${escapeHtml(company?.addr1 || "")}</div>
                <div class="small">${escapeHtml(company?.addr2 || "")}</div>
                <div class="small">${escapeHtml(company?.city || "")}</div>
                <div class="small">State: ${escapeHtml(
                  company?.state || "-"
                )}, Code: ${escapeHtml(company?.stateCode || "-")}</div>
                <div class="small">GSTIN/UIN: ${escapeHtml(
                  company?.gstin || "-"
                )}</div>
                ${
                  company?.email
                    ? `<div class="small">E-Mail : ${escapeHtml(
                        company.email
                      )}</div>`
                    : ""
                }
                ${
                  company?.mobile || company?.phone || company?.whatsapp
                    ? `<div class="small">Phone : ${escapeHtml(
                        [company.mobile, company.phone]
                          .filter(Boolean)
                          .join(" / ")
                      )}${
                        company?.whatsapp
                          ? " | WhatsApp: " + escapeHtml(company.whatsapp)
                          : ""
                      }</div>`
                    : ""
                }
              </div>
              <div style="font-size:12px;font-weight:700">Tax invoice</div>
            </div>

            <div class="two">
              <div class="box">
                <div style="font-weight:600;font-size:12px">Consignee (Ship To)</div>
                <div style="font-weight:600">${escapeHtml(
                  useConsignee.name || "-"
                )}</div>
                
                <div class="small">${escapeHtml(
                  useConsignee.addr1 || ""
                )}</div>
                <div class="small">${escapeHtml(
                  useConsignee.addr2 || ""
                )}</div>
                <div class="small">${escapeHtml(
                  useConsignee.city || ""
                )}</div>
                <div class="small">State: ${escapeHtml(
                  useConsignee.state || "-"
                )}, Code: ${escapeHtml(useConsignee.stateCode || "-")}</div>
                <div class="small">GSTIN/UIN: ${escapeHtml(
                  useConsignee.gstin || "-"
                )}</div>
              </div>
              <div class="box">
                <div style="font-weight:600;font-size:12px">Buyer (Bill To)</div>
                <div style="font-weight:600">${escapeHtml(
                  buyer.name || "-"
                )}</div>
                  ${
    buyer.mobile
      ? `<div class="small">Mobile: ${escapeHtml(buyer.mobile)}</div>`
      : ""
  }
                
                <div class="small">${escapeHtml(buyer.addr1 || "")}</div>
                <div class="small">${escapeHtml(buyer.addr2 || "")}</div>
                <div class="small">${escapeHtml(buyer.city || "")}</div>
                <div class="small">State: ${escapeHtml(
                  buyer.state || "-"
                )}, Code: ${escapeHtml(buyer.stateCode || "-")}</div>
                <div class="small">GSTIN/UIN: ${escapeHtml(
                  buyer.gstin || "-"
                )}</div>
              </div>
            </div>
          </div>

          <div class="box">
            <table>
              <tbody>
                <tr><td style="width:45%">Invoice No</td><td>${escapeHtml(
                  invMeta.invoiceNo
                )}</td></tr>
                <tr><td>Date</td><td>${fmtL(invMeta.date)}</td></tr>
                <tr><td>Terms of Payment</td><td>${escapeHtml(
                  paymentMode || "-"
                )}</td></tr>
                <tr>
                  <td>Payment Receipt No</td>
                  <td>${escapeHtml(paymentReceiptNo || "-")}</td>
                </tr>
                <tr><td>e-Way Bill No</td><td>${escapeHtml(
                  invMeta.ewayNo || "-"
                )}</td></tr>
                <tr><td>LR / Date</td><td>${escapeHtml(
                  invMeta.lrDate || "-"
                )}</td></tr>
                <tr><td>Vehicle No</td><td>${escapeHtml(
                  invMeta.vehicleNo || "-"
                )}</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <table style="margin-top:8px">
          <thead>
            <tr>
              <th style="width:36px">S.No</th>
              <th>Description of Goods</th>
              <th>HSN/SAC</th>
              <th class="r">Quantity</th>
              <th>Unit</th>
              <th class="r">Rate</th>
              <th class="r">Price</th>
              <th class="c">GST %</th>
              <th class="r">Amount</th>
            </tr>
          </thead>

          <tbody>
            ${rowsHtml}
            <tr><td colspan="8" class="r"><b>Taxable Value</b></td><td class="r"><b>${toINR(
              computed.taxable
            )}</b></td></tr>
            <tr><td colspan="8" class="r">Discount (${Number(
              discountPct || 0
            )}%)</td><td class="r">-${toINR(
      computed.discountAmount
    )}</td></tr>
            <tr><td colspan="8" class="r">Taxable After Discount</td><td class="r">${toINR(
              computed.discountedTaxable
            )}</td></tr>
            ${
              isInterState
                ? `<tr><td colspan="8" class="r"><b>IGST (rounded)</b></td><td class="r"><b>${toINR(
                    computed.igstRounded
                  )}</b></td></tr>`
                : `<tr><td colspan="8" class="r">CGST (rounded)</td><td class="r">${toINR(
                    computed.cgstRounded
                  )}</td></tr>
                   <tr><td colspan="8" class="r">SGST (rounded)</td><td class="r">${toINR(
                     computed.sgstRounded
                   )}</td></tr>`
            }
            <tr><td colspan="8" class="r">Round Off (GST rounding)</td><td class="r">${toINR(
              computed.roundOff
            )}</td></tr>
            <tr><td colspan="8" class="r"><b>Total</b></td><td class="r"><b>${toINR(
              computed.total
            )}</b></td></tr>
            <tr><td colspan="8" class="r"><b>Grand Total</b></td><td class="r"><b>${toINR(
              computed.grand
            )}</b></td></tr>

            ${splitLinesHtml}

            <tr><td colspan="8" class="r">Advance / Received</td><td class="r">-${toINR(
              adv
            )}</td></tr>
            <tr><td colspan="8" class="r"><b>Balance Due</b></td><td class="r"><b>${toINR(
              balanceDue
            )}</b></td></tr>
          </tbody>
        </table>

        <div class="foot">
          <div class="box">
            <div><b>Amount Chargeable (in words)</b></div>
            <div class="small" style="margin-top:4px">INR ${numberToWordsIndian(
              Math.round(computed.grand)
            )} Only</div>
            <div class="hr"></div>
            <div class="small"><b>Tax Amount (in words)</b></div>
            <div class="small">INR ${numberToWordsIndian(
              Math.round(computed.gstRounded)
            )} Only</div>
          </div>
          <div class="box">
            <div class="small"><b>Company's Bank Details</b></div>
            ${
              defaultBank
                ? `
              <div class="small">Bank : ${escapeHtml(
                defaultBank.bankName || ""
              )}</div>
              <div class="small">Branch & IFSC : ${escapeHtml(
                defaultBank.branch || ""
              )} & ${escapeHtml(
                    defaultBank.ifsc || ""
                  )}</div>
              <div class="small">A/C No : ${escapeHtml(
                defaultBank.account || ""
              )}</div>`
                : `<div class="small">No bank selected</div>`
            }
            <div class="hr"></div>
            <div class="small c">For ${escapeHtml(company?.name || "")}</div>
            <div style="height:48px"></div>
            <div class="small r">Authorised Signatory</div>
          </div>
        </div>

        <div class="small" style="margin-top:8px">
          <b>Declaration:</b> We declare that this invoice shows the actual price of the goods described and that all particulars are true and correct.
          <div>SUBJECT TO TRICHY JURISDICTION</div>
        </div>
      </div>
      <script>window.onload = () => window.print();</script>
    </body></html>`;

    wnd.document.write(html);
    wnd.document.close();
  };

  // ================= UI =================
  return (
    <div className="billing-full card">
      <div className="row">
        {/* Invoice meta */}
        <div>
          <div className="form-grid-4" style={{ marginTop: 6 }}>
          <div className="form-grid-3" style={{ marginTop: 6 }}>
  {/* Invoice Type */}
  <div>
    <label>Invoice Type</label>
    <select
      value={invoiceType}
      onChange={(e) => setInvoiceType(e.target.value)}
    >
      <option value="B2C">B2C</option>
      <option value="B2B">B2B</option>
    </select>
  </div>

  {/* Date */}
  <div>
    <label>Date</label>
    <input
      type="date"
      value={invMeta.date}
      onChange={(e) =>
        setInvMeta((m) => ({ ...m, date: e.target.value }))
      }
    />
  </div>
</div>

            <div
              style={{ display: "flex", gap: "20px", alignItems: "center" }}
            >
              <div className="meta-row">
                <div className="meta-field">
                  <label>eWay Bill No</label>
                  <input
                    value={invMeta.ewayNo}
                    onChange={(e) =>
                      setInvMeta((m) => ({ ...m, ewayNo: e.target.value }))
                    }
                  />
                </div>
                <div className="meta-field">
  <             label>Vehicle No</label>
                  <input
                    value={invMeta.vehicleNo}
                    onChange={(e) =>
                      setInvMeta((m) => ({ ...m, vehicleNo: e.target.value.toUpperCase() }))
                    }
                    placeholder="Vehicle No."
                  />
                </div>


                <div className="meta-field">
                  <label>KMs</label>
                  <input
                    value={invMeta.kms || ""}
                    onChange={(e) =>
                      setInvMeta((m) => ({
                        ...m,
                        kms: e.target.value.replace(/[^0-9]/g, ""),
                      }))
                    }
                  />
                </div>

                <div className="meta-field">
                  <label>From Date</label>
                  <input
                    type="date"
                    value={invMeta.fromDate || ""}
                    onChange={(e) =>
                      setInvMeta((m) => ({ ...m, fromDate: e.target.value }))
                    }
                  />
                </div>

                <div className="meta-field">
                  <label>To Date</label>
                  <input
                    type="date"
                    value={invMeta.toDate || ""}
                    onChange={(e) =>
                      setInvMeta((m) => ({ ...m, toDate: e.target.value }))
                    }
                  />
                </div>
              </div>
            </div>
            {/*
            <div>
              <label>LR Date</label>
              <input
                type="date"
                value={invMeta.lrDate}
                onChange={(e) =>
                  setInvMeta((m) => ({ ...m, lrDate: e.target.value }))
                }
              />
            </div>
            */}
          </div>
        </div>

        {/* Buyer */}
        <div>
          <div className="small">Buyer (Bill To)</div>
          <div className="form-grid-4" style={{ marginTop: 6 }}>
           <div className="row-2">
  <div style={{ position: "relative" }}>
  <label>Name *</label>
  <input
    value={buyer.name}
    onChange={(e) => {
      const v = e.target.value;
      setBuyer((b) => ({ ...b, name: v }));
      setNameQuery(v);
    }}
    onFocus={() => setNameQuery(buyer.name || " ")}
    placeholder="Type buyer name..."
    autoComplete="off"
  />

  {/* dropdown */}
  {nameSuggest.length > 0 && (
    <div
      style={{
        position: "absolute",
        top: "100%",
        left: 0,
        right: 0,
        background: "#fff",
        border: "1px solid #ddd",
        zIndex: 50,
        maxHeight: 220,
        overflowY: "auto",
        borderRadius: 8,
      }}
    >
      {nameSuggest.map((a, idx) => (
        <div
          key={`${a.mobile || ""}-${idx}`}
          onMouseDown={(e) => {
            e.preventDefault(); // IMPORTANT: prevents blur before click
            pickName(a);
          }}
          style={{
            padding: "8px 10px",
            cursor: "pointer",
            borderBottom: "1px solid #f1f1f1",
          }}
        >
          <div style={{ fontWeight: 600 }}>{a.name}</div>
          <div style={{ fontSize: 12, color: "#666" }}>
            {a.mobile ? `📱 ${a.mobile}` : ""}
            {a.city ? ` • ${a.city}` : ""}
          </div>
        </div>
      ))}
    </div>
  )}
</div>
<div>
  <label>Mobile No.</label>
  <input
    type="tel"
    list="mobile-suggest"
    placeholder="10-digit mobile"
    value={buyer.mobile}
    onChange={async (e) => {
      const v = String(e.target.value).replace(/[^0-9]/g, "").slice(0, 10);

      // update buyer mobile immediately
      setBuyer((b) => ({ ...b, mobile: v }));
      setMobileQuery(v);

      // reset ledger if user edits mobile again
      if (v.length < 10) {
        setSelectedAccount(null);
        setAccQuery("");
        return;
      }

      // when full 10 digits → lookup
      if (v.length === 10) {
        try {
          const acc = await apiAccountByMobile(v);
          console.log("Lookup by mobile:", v, acc);

          if (acc) {
            const normalized = {
              ...acc,
              name: acc.name ?? acc.account_name ?? "",
            };

            // 1️⃣ Fill buyer details
            fillBuyerFromAccount(normalized);

            // 2️⃣ Auto-assign ledger
            setSelectedAccount(normalized);
            setAccQuery(normalized.name);

            // 3️⃣ Close all suggestions
            setAccSuggest([]);
            setMobileSuggest([]);
          }
        } catch (err) {
          console.error("Mobile lookup failed", err);
        }
      }
    }}
    onBlur={async () => {
      const v = (buyer.mobile || "").trim();

      if (v.length === 10 && !selectedAccount) {
        try {
          const acc = await apiAccountByMobile(v);

          if (acc) {
            const normalized = {
              ...acc,
              name: acc.name ?? acc.account_name ?? "",
            };

            fillBuyerFromAccount(normalized);
            setSelectedAccount(normalized);
            setAccQuery(normalized.name);
          }
        } catch {}
      }

      setMobileSuggest([]);
    }}
  />

  <datalist id="mobile-suggest">
    {mobileSuggest.map((a, idx) => (
      <option key={`${a.mobile}-${idx}`} value={a.mobile}>
        {a.name ? `${a.name} - ${a.mobile}` : a.mobile}
      </option>
    ))}
  </datalist>
</div>  
</div>

           <div className="row-2">
  {/* GSTIN */}
  <div>
    <label>GSTIN / UIN</label>
    <input
      value={buyer.gstin}
      onChange={(e) =>
        setBuyer((b) => ({ ...b, gstin: e.target.value }))
      }
    />
  </div>

  {/* ACCOUNT (LEDGER) */}
  <div style={{ position: "relative" }}>
    <label>Account (Ledger)</label>
    <input
      value={accQuery}
       
        tabIndex={-1}   
        readOnly={!!selectedAccount}
      onChange={(e) => {
        setAccQuery(e.target.value);
      //  setSelectedAccount(null);
      }}
      onFocus={() => {
  if (accQuery.length === 0) setAccSuggest([]);
}}
      placeholder="Search account…"
      autoComplete="off"
    />

    {/* Dropdown */}
    {accSuggest.length > 0 && (
      <div
        style={{
          position: "absolute",
          top: "100%",
          left: 0,
          right: 0,
          background: "#fff",
          border: "1px solid #ddd",
          zIndex: 60,
          maxHeight: 220,
          overflowY: "auto",
          borderRadius: 8,
        }}
      >
        {accSuggest.map((a, idx) => (
          <div
            key={`${a.account_id}-${idx}`}
            onMouseDown={(e) => {
              e.preventDefault(); // prevent blur
              pickAccount(a);
            }}
            style={{
              padding: "8px 10px",
              cursor: "pointer",
              borderBottom: "1px solid #f1f1f1",
            }}
          >
            <div style={{ fontWeight: 600 }}>{a.name}</div>
            <div style={{ fontSize: 12, color: "#666" }}>
              ID: {a.account_id}
              {a.mobile ? ` • 📱 ${a.mobile}` : ""}
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
</div>

            <div className="row-2">
              <div>
                <label>State</label>
                <input
                  value={buyer.state}
                  onChange={(e) =>
                    setBuyer((b) => ({ ...b, state: e.target.value }))
                  }
                  placeholder="Tamil Nadu"
                />
              </div>
              <div>
                <label>State Code</label>
                <input
                  value={buyer.stateCode}
                  onChange={(e) =>
                    setBuyer((b) => ({
                      ...b,
                      stateCode: String(e.target.value).replace(/[^0-9]/g, ""),
                    }))
                  }
                  placeholder="33"
                />
              </div>
            </div>

            <div>
              <label>Address 1 *</label>
              <input
                value={buyer.addr1}
                onChange={(e) =>
                  setBuyer((b) => ({ ...b, addr1: e.target.value }))
                }
              />
            </div>
            <div>
              <label>Address 2</label>
              <input
                value={buyer.addr2}
                onChange={(e) =>
                  setBuyer((b) => ({ ...b, addr2: e.target.value }))
                }
              />
            </div>
            <div className="row-2">
              <div>
                <label>City *</label>
                <input
                  value={buyer.city}
                  onChange={(e) =>
                    setBuyer((b) => ({ ...b, city: e.target.value }))
                  }
                />
              </div>
              <div>
                <label>PIN *</label>
                <input
                  value={buyer.pin}
                  onChange={(e) =>
                    setBuyer((b) => ({
                      ...b,
                      pin: String(e.target.value).replace(/[^0-9]/g, ""),
                    }))
                  }
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Consignee toggle */}
      <div className="stack" style={{ marginTop: 4 }}>
        <label
          className="small"
          style={{ display: "flex", alignItems: "center", gap: 8 }}
        >
          <input
            type="checkbox"
            checked={consigneeDifferent}
            onChange={(e) => setConsigneeDifferent(e.target.checked)}
            style={{ width: 16, height: 16 }}
          />
          Consignee different from Buyer?
        </label>
        {!consigneeDifferent && (
          <span className="badge">Using Buyer as Consignee</span>
        )}
      </div>

      {/* Consignee fields */}
      {consigneeDifferent && (
        <div className="card" style={{ marginTop: 8 }}>
          <div className="small">Consignee (Ship To)</div>

          <div className="form-grid-3" style={{ marginTop: 6 }}>
            <div>
              <label>Name</label>
              <input
                value={consignee.name}
                onChange={(e) =>
                  setConsignee((c) => ({ ...c, name: e.target.value }))
                }
              />
            </div>
            <div>
              <label>GSTIN/UIN</label>
              <input
                value={consignee.gstin}
                onChange={(e) =>
                  setConsignee((c) => ({ ...c, gstin: e.target.value }))
                }
              />
            </div>
            <div>
              <label>State Code</label>
              <input
                value={consignee.stateCode}
                onChange={(e) =>
                  setConsignee((c) => ({
                    ...c,
                    stateCode: String(e.target.value).replace(/[^0-9]/g, ""),
                  }))
                }
                placeholder="33"
              />
            </div>
          </div>

          <div className="form-grid-3" style={{ marginTop: 6 }}>
            <div>
              <label>State</label>
              <input
                value={consignee.state}
                onChange={(e) =>
                  setConsignee((c) => ({ ...c, state: e.target.value }))
                }
                placeholder="Tamil Nadu"
              />
            </div>
            <div>
              <label>City</label>
              <input
                value={consignee.city}
                onChange={(e) =>
                  setConsignee((c) => ({ ...c, city: e.target.value }))
                }
              />
            </div>
            <div>
              <label>PIN</label>
              <input
                value={consignee.pin || ""}
                onChange={(e) =>
                  setConsignee((c) => ({
                    ...c,
                    pin: String(e.target.value).replace(/[^0-9]/g, ""),
                  }))
                }
              />
            </div>
          </div>

          <div style={{ marginTop: 6 }}>
            <label>Address 1</label>
            <input
              value={consignee.addr1}
              onChange={(e) =>
                setConsignee((c) => ({ ...c, addr1: e.target.value }))
              }
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ marginTop: 6 }}>
            <label>Address 2</label>
            <input
              value={consignee.addr2}
              onChange={(e) =>
                setConsignee((c) => ({ ...c, addr2: e.target.value }))
              }
              style={{ width: "100%" }}
            />
          </div>
        </div>
      )}

      {/* Items table */}
      <div className="card" style={{ marginTop: 12 }}>
        <table
          className="items-table"
          style={{ width: "100%", tableLayout: "fixed" }}
        >
          <colgroup>
            <col style={{ width: 44 }} />
            <col />
            <col style={{ width: 100 }} />
            <col style={{ width: 84 }} />
            <col style={{ width: 72 }} />
            <col style={{ width: 96 }} />
            <col style={{ width: 108 }} />
            <col style={{ width: 84 }} />
            <col style={{ width: 116 }} />
            <col style={{ width: 56 }} />
          </colgroup>

          <thead>
            <tr>
              <th>#</th>
              <th>Description of Goods</th>
              <th>HSN</th>
              <th className="right">Qty</th>
              <th className="center">Unit</th>
              <th className="right">Rate</th>
              <th className="right">Price</th>
              <th className="right">GST %</th>
              <th className="right">Amount</th>
              <th className="center">✖</th>
            </tr>
          </thead>

          <tbody>
            {lines.map((l, i) => {
              const qty = Number(l.qty || 0);
              const rate = Number(l.unitPrice || 0);
              const gstPct = Number(l.gst || 0);
              const price = qty * rate;
              const st = statuses[i];

              const disableRowInputs =
                !!st &&
                (!st.exists ||
                  (typeof st.stock === "number" && st.stock <= 0));

              return (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    <input
                      list="bill-suggest"
                      placeholder="Type to search Purchases…"
                      value={l.itemName}
                      onChange={(e) => handleItemChange(i, e.target.value)}
                      onFocus={(e) =>
                        setActiveQuery(e.target.value || " ")
                      }
                    />
                    <datalist id="bill-suggest">
                      {suggest.map((s, idx) => (
                        <option
                          key={`${s.name}-${idx}`}
                          value={s.name}
                        />
                      ))}
                    </datalist>
                    {l.itemName && st && (
                      <div
                        className="small"
                        style={{
                          color:
                            st.exists &&
                            Number(l.qty || 0) <= st.stock
                              ? "#047857"
                              : "#b91c1c",
                        }}
                      >
                        {!st.exists && "Not available in Inventory"}
                        {st.exists &&
                          st.stock === 0 &&
                          "Out of stock"}
                        {st.exists &&
                          st.stock > 0 &&
                          Number(l.qty || 0) === 0 &&
                          `In stock: ${st.stock}`}
                        {st.exists &&
                          Number(l.qty || 0) > st.stock &&
                          `Only ${st.stock} in stock`}
                        {st.exists &&
                          Number(l.qty || 0) > 0 &&
                          Number(l.qty || 0) <= st.stock &&
                          `Available: ${st.stock}`}
                      </div>
                    )}
                  </td>

                  <td>
                    <input
                      value={l.hsn}
                      onChange={(e) => handleHSN(i, e.target.value)}
                      disabled={disableRowInputs}
                    />
                  </td>

                  <td className="right">
                    <input
                      value={l.qty}
                      onChange={(e) => handleQty(i, e.target.value)}
                      onBlur={(e) =>
                        updateAvailability(
                          i,
                          l.itemName,
                          e.target.value
                        )
                      }
                      disabled={disableRowInputs}
                    />
                  </td>

                  <td className="center">
                    <input value="KG" readOnly />
                  </td>

                  <td className="right">
                    <input
                      value={l.unitPrice}
                      onChange={(e) => handlePrice(i, e.target.value)}
                      disabled={disableRowInputs}
                    />
                  </td>

                  <td className="right">{toINR(price)}</td>

                  <td className="right">
                    <input
                      value={l.gst}
                      onChange={(e) => handleGST(i, e.target.value)}
                      disabled={disableRowInputs}
                    />
                  </td>

                  <td className="right">
                    {toINR(price + (price * gstPct) / 100)}
                  </td>

                  <td className="center">
                    <button
                      className="btn warn"
                      onClick={() => removeRow(i)}
                    >
                      Del
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>

          <tfoot>
            <tr>
              <td colSpan={10}>
                <div className="stack">
                  <button className="btn" onClick={addRow}>
                    + Add Item
                  </button>
                  <div className="muted">
                    Type to search from <b>Purchases</b>. Unit is fixed
                    to <b>KG</b>; Rate &amp; GST% auto-fill from the
                    latest Purchase; HSN from Inventory.
                  </div>
                </div>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Totals + payments */}
      <div className="totals-grid">
        <div />
        <div className="card" style={{ padding: 12 }}>
          <div
            className="row"
            style={{ gridTemplateColumns: "1fr 1fr", margin: 0 }}
          >
            <div className="small">Taxable Amount</div>
            <div className="right">
              <b>{toINR(computed.taxable)}</b>
            </div>

            <div
              className="small"
              style={{ display: "flex", alignItems: "center", gap: 8 }}
            >
              Discount (%)
            </div>
            <div className="right">
              <input
                value={discountPct}
                onChange={(e) =>
                  setDiscountPct(
                    String(e.target.value).replace(/[^0-9.]/g, "")
                  )
                }
                style={{ width: 120, textAlign: "right" }}
                placeholder="0"
              />
            </div>

            <div className="small">Discount Amount</div>
            <div className="right">
              -{toINR(computed.discountAmount)}
            </div>

            <div className="small">
              {isInterState ? "GST (IGST)" : "GST (CGST+SGST)"}
            </div>
            <div className="right">
              <b>
                {isInterState
                  ? toINR(computed.igstRounded)
                  : toINR(
                      computed.cgstRounded + computed.sgstRounded
                    )}
              </b>
            </div>

            <div className="small">Round Off (GST rounding)</div>
            <div className="right">
              {toINR(computed.roundOff)}
            </div>

            <div className="small">Total</div>
            <div className="right">
              <b>{toINR(computed.total)}</b>
            </div>

            <div className="small" style={{ marginTop: 6 }}>
              <b>Grand Total</b>
            </div>
            <div className="right" style={{ marginTop: 6 }}>
              <b>{toINR(computed.grand)}</b>
            </div>

            {/* Advance / Partial Payment field hidden */}
            {/* 
            <div className="small" style={{ marginTop: 8 }}>
              Advance / Partial Payment
            </div>
            <div className="right" style={{ marginTop: 8 }}>
              <input
                value={advance}
                onChange={(e) =>
                  setAdvance(
                    String(e.target.value).replace(/[^0-9]/g, "")
                  )
                }
                style={{ width: 120, textAlign: "right" }}
                placeholder="0"
              />
            </div>
            */}

            <div className="small">Terms of Payment</div>
            <div className="right">
              <select
                value={paymentMode}
                onChange={(e) => setPaymentMode(e.target.value)}
                style={{ width: 140, textAlign: "right" }}
              >
                <option value="">Select</option>
                {paymentModes.map((m) => (
                  <option key={"pm-" + m}>{m}</option>
                ))}
              </select>
            </div>
          <div className="small">Payment Receipt No.</div>
          <div className="right">
            <textarea
              value={paymentReceiptNo}
              onChange={(e) => setPaymentReceiptNo(e.target.value)}
              placeholder="Enter receipt / UTR / cheque / reference number"
              rows={2}
              style={{
                width: "100%",
                resize: "vertical",
                textAlign: "left",
              }}
            />
          </div>

          </div>

          {/* Split Payments */}
          <hr />
          <div
            className="row"
            style={{ gridTemplateColumns: "1fr 1fr", marginTop: 6 }}
          >
            <div
              className="small"
              style={{ display: "flex", alignItems: "center", gap: 8 }}
            >
              Split Payment (max 2)
            </div>
            <div className="right">
              <label
                style={{
                  display: "inline-flex",
                  gap: 8,
                  alignItems: "center",
                }}
              >
                <input
                  type="checkbox"
                  checked={splitEnabled}
                  onChange={(e) => setSplitEnabled(e.target.checked)}
                />
                <span className="small">Enable</span>
              </label>
            </div>
          </div>

          {splitEnabled && (
            <>
              <div
                className="grid grid-3"
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr 1fr",
                  gap: 8,
                  marginTop: 8,
                }}
              >
                <div>
                  <label>Split #1 Amount</label>
                  <input
                    type="number"
                    step="0.01"
                    value={split1.amount}
                    onChange={(e) => {
                      const raw = String(e.target.value);
                      const clean = raw.replace(/[^0-9.]/g, "");
                      const amt1 = n(clean);
                      const remaining = Math.max(
                        0,
                        +(computed.grand - amt1).toFixed(2)
                      );
                      setSplit1({ ...split1, amount: clean });
                      setSplit2((prev) => ({
                        ...prev,
                        amount:
                          remaining > 0
                            ? String(remaining)
                            : remaining === 0
                            ? "0"
                            : prev.amount,
                      }));
                    }}
                    placeholder="0"
                    className="right"
                  />
                </div>
                <div>
                  <label>Split #1 Terms / Mode</label>
                  <select
                    value={split1.mode}
                    onChange={(e) =>
                      setSplit1({ ...split1, mode: e.target.value })
                    }
                  >
                    {paymentModes.map((m) => (
                      <option key={"m1" + m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label>Ref / Txn No. (opt)</label>
                  <input
                    value={split1.ref}
                    onChange={(e) =>
                      setSplit1({ ...split1, ref: e.target.value })
                    }
                    placeholder="UTR / Cheque / Last 4"
                  />
                </div>
              </div>

              <div
                className="grid grid-3"
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr 1fr",
                  gap: 8,
                  marginTop: 8,
                }}
              >
                <div>
                  <label>Split #2 Amount</label>
                  <input
                    type="number"
                    step="0.01"
                    value={split2.amount}
                    onChange={(e) =>
                      setSplit2({ ...split2, amount: e.target.value })
                    }
                    placeholder="0"
                    className="right"
                  />
                </div>
                <div>
                  <label>Split #2 Terms / Mode</label>
                  <select
                    value={split2.mode}
                    onChange={(e) =>
                      setSplit2({ ...split2, mode: e.target.value })
                    }
                  >
                    {paymentModes.map((m) => (
                      <option key={"m2" + m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label>Ref / Txn No. (opt)</label>
                  <input
                    value={split2.ref}
                    onChange={(e) =>
                      setSplit2({ ...split2, ref: e.target.value })
                    }
                    placeholder="UTR / Cheque / Last 4"
                  />
                </div>
              </div>

              <div
                className="row"
                style={{
                  gridTemplateColumns: "1fr 1fr",
                  marginTop: 8,
                }}
              >
                <div className="small">
                  Paid Now (Advance + Splits)
                </div>
                <div className="right">{toINR(paidNow)}</div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* SAVE & PRINT */}
      <div className="actions-row">
        <button
          className="btn"
          onClick={async () => {
            const missing = [];
            if (!buyer.name.trim()) missing.push("Buyer Name");
            if (!buyer.addr1.trim()) missing.push("Buyer Address 1");
            if (!buyer.city.trim()) missing.push("Buyer City");
            if (!buyer.pin?.trim?.()) missing.push("Buyer PIN");
            console.log("Selected Account:", selectedAccount);
            if (!selectedAccount?.account_id) {
              alert("Please select Account (Ledger) before saving.");
            return;
  }


            if (missing.length) {
              alert(
                "Please fill mandatory fields:\n" + missing.join("\n")
              );
              return;
            }

            const issues = [];
            lines.forEach((l, i) => {
              if (!l.itemName || !Number(l.qty)) return;
              const st = statuses[i];
              if (!st?.exists)
                issues.push(
                  `Row ${i + 1}: Item not found in inventory`
                );
              else if (!st?.can_deliver)
                issues.push(
                  `Row ${i + 1}: Only ${st.stock} in stock`
                );
            });
            if (issues.length) {
              alert("Please fix before saving:\n" + issues.join("\n"));
              return;
            }

            if (!computed.rows.length) {
              alert("No valid line items to save.");
              return;
            }

const salePayload = {
  bill_no: invMeta.invoiceNo,
  date: invMeta.date,
  account_id: selectedAccount.account_id,
  party_name: buyer.name || "",
  invoice_type: invoiceType,

  // 🚚 TRANSPORT / E-WAY DETAILS (ADD THIS)
  eway_no: invMeta.ewayNo || "",
  vehicle_no: invMeta.vehicleNo || "",
  kms: invMeta.kms ? Number(invMeta.kms) : null,
  from_date: invMeta.fromDate || null,
  to_date: invMeta.toDate || null,

  // totals
  taxable: +computed.taxable.toFixed(2),
  discount_amount: +computed.discountAmount.toFixed(2),
  discounted_taxable: +computed.discountedTaxable.toFixed(2),
  gst_amount: +computed.gstRounded.toFixed(2),
  round_off: +computed.roundOff.toFixed(2),
  grand_total: +computed.grand.toFixed(2),

  paid_now: +paidNow.toFixed(2),
  balance: +balanceAfterSplits.toFixed(2),

  payment_mode: paymentMode || "",
  payment_receipt_no: paymentReceiptNo || "",

  // items
  items: computed.rows.map((r) => ({
    item_name: r.itemName,
    hsn: r.hsn || "",
    qty: Number(r.qty || 0),
    unit: r.unit || "KG",
    price: Number(r.rate || 0),
    gst: Number(r.gstRate || r.gst || 0),
  })),
};



            try {
              const resp = await apiCreateSaleFIFO(salePayload);
              // Optional: you can log / show resp.sale_id, resp.total_amount, etc.
              console.log("Sale created (FIFO):", resp);
            } catch (e) {
              console.error("Error creating sale via FIFO:", e);
              alert(
                "Failed to create sale in database (FIFO).\n" +
                  (e?.message || e)
              );
              return;
            }

            // 2) Local storage copy (for your own dashboard/history)
            const bill = {
              id: Date.now(),
              account_id: selectedAccount.account_id,
              account_name: selectedAccount.name,
              account_mobile: selectedAccount.mobile || buyer.mobile || "",
              company,
              buyer,
              consignee: consigneeDifferent ? consignee : buyer,
              invMeta,
              lines,
              computed,
              discountPct,
              advance,
              paymentMode,
              payments: {
                splitEnabled,
                split1: splitEnabled
                  ? {
                      amount: n(split1.amount),
                      mode: split1.mode,
                      ref: split1.ref,
                    }
                  : null,
                split2: splitEnabled
                  ? {
                      amount: n(split2.amount),
                      mode: split2.mode,
                      ref: split2.ref,
                    }
                  : null,
                paidNow,
                balance: balanceAfterSplits,
              },
              createdAt: new Date().toISOString(),
            };
            const arr = JSON.parse(
              localStorage.getItem("bills.v2") || "[]"
            );
            arr.push(bill);
            localStorage.setItem("bills.v2", JSON.stringify(arr));

            // 3) Export + Print
            exportExcel();
            printNow();
          }}
        >
          Save
        </button>

        <button className="btn primary" onClick={printNow}>
          Print
        </button>
      </div>
    </div>
  );
}
