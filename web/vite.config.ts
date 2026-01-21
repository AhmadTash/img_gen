import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy so the React app can call the FastAPI server without CORS hassle.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Listen on all addresses (0.0.0.0)
    proxy: {
      "/generate": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
