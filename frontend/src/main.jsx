import React from "react";
import { createRoot } from "react-dom/client";
import IACTDashboard from "@iact";

// Empty string = same origin (frontend served by the FastAPI backend).
// Undefined = local development against http://127.0.0.1:8000.
window.IACT_API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <IACTDashboard />
  </React.StrictMode>
);
