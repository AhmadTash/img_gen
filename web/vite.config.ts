import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy so the React app can call the FastAPI server without CORS hassle.
// In production, VITE_API_URL will be used directly in the frontend code.
const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Listen on all addresses (0.0.0.0)
    proxy: {
      "/generate": {
        target: apiUrl,
        changeOrigin: true,
      },
      "/health": {
        target: apiUrl,
        changeOrigin: true,
      },
      "/log-feedback": {
        target: apiUrl,
        changeOrigin: true,
      },
      "/suggest-params": {
        target: apiUrl,
        changeOrigin: true,
      },
    },
  },
});
