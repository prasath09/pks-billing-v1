import React, { useState } from "react";
import B2CReturns from "./B2CReturns";
// future-ready (not used yet)
// import B2BReturns from "./B2BReturns";
import PurchaseReturns from "./PurchaseReturns";

export default function ReturnsPanel() {
  const [subTab, setSubTab] = useState("btc");

  return (
    <div className="card">
      <div className="small">Returns</div>

      {/* Sub Tabs */}
      <div className="tabs" style={{ marginTop: 8 }}>
        <button
          className={`tab ${subTab === "btc" ? "active" : ""}`}
          onClick={() => setSubTab("btc")}
        >
          B to C
        </button>

        {/* <button
          className={`tab ${subTab === "btb" ? "active" : ""}`}
          onClick={() => setSubTab("btb")}
        >
          B to B
        </button> */}

        <button
          className={`tab ${subTab === "purchase" ? "active" : ""}`}
          onClick={() => setSubTab("purchase")}
        >
          Purchase
        </button>
      </div>

      {/* Content */}
      <div style={{ marginTop: 12 }}>
        {subTab === "btc" && <B2CReturns />}

        {/* {subTab === "btb" && <B2CReturns />} */}
        
        {subTab === "purchase" && <PurchaseReturns />} 
    
      </div>
    </div>
  );
}
