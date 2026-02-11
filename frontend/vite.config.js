import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      "/predict": "http://localhost:5000",
      "/analyze": "http://localhost:5000",
      "/health": "http://localhost:5000",
      "/evaluation": "http://localhost:5000",
    },
  },
});
