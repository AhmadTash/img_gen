import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

// Prevent zooming via wheel (Ctrl + Scroll)
document.addEventListener('wheel', (e) => {
  if (e.ctrlKey) e.preventDefault();
}, { passive: false });

// Prevent zooming via keyboard (Ctrl/Cmd + +/-/0)
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && (['+', '-', '=', '0'].includes(e.key))) {
    e.preventDefault();
  }
});

// Prevent pinch-to-zoom gestures
document.addEventListener('touchstart', (e) => {
  if (e.touches.length > 1) e.preventDefault();
}, { passive: false });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
