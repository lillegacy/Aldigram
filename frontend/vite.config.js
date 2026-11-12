import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Single source of truth: the component at the repository root.
      "@iact": path.resolve(here, "../IACTDashboard.jsx"),
    },
  },
  server: {
    host: true,
    port: 5173,
    fs: {
      allow: [path.resolve(here, "..")],
    },
  },
});
