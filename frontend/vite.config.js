import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// The component lives at the repository root, outside this Vite project.
// Alias it (and its React imports) to this project's node_modules so both the
// dev server and production builds resolve everything consistently.
const alias = [
  { find: "@iact", replacement: path.resolve(here, "../IACTDashboard.jsx") },
  {
    find: /^react$/,
    replacement: path.resolve(here, "node_modules/react/index.js"),
  },
  {
    find: /^react\/jsx-runtime$/,
    replacement: path.resolve(here, "node_modules/react/jsx-runtime.js"),
  },
  {
    find: /^react\/jsx-dev-runtime$/,
    replacement: path.resolve(here, "node_modules/react/jsx-dev-runtime.js"),
  },
  {
    find: /^react-dom$/,
    replacement: path.resolve(here, "node_modules/react-dom/index.js"),
  },
  {
    find: /^react-dom\/client$/,
    replacement: path.resolve(here, "node_modules/react-dom/client.js"),
  },
];

export default defineConfig({
  plugins: [react()],
  resolve: { alias },
  server: {
    host: true,
    port: 5173,
    fs: {
      allow: [path.resolve(here, "..")],
    },
  },
});
