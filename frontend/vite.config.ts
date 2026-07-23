import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built app is served by the FastAPI console at /app (see
// `console/app.py`'s conditional StaticFiles mount over `frontend/dist`),
// so `base` must match that mount point for asset URLs to resolve once
// built. In dev, `server.proxy` forwards /api to the console backend
// running separately on :8000.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
