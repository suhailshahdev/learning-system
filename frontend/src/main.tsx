import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";

import { ThemeProvider } from "@/components/theme-provider";
import App from "./App.tsx";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Localhost backend is fast and reliable enough that stock "always
      // stale" is wasteful, especially under dev HMR. 30s gives cached
      // data a reasonable window while staying fresh enough.
      staleTime: 30_000,
      // Localhost failures usually mean the backend is down. Retrying
      // three times just delays the inevitable error message.
      retry: 1,
    },
  },
});

const rootElement = document.getElementById("root");
if (rootElement === null) {
  throw new Error("Root element #root not found in index.html");
}

createRoot(rootElement).render(
  <StrictMode>
    <BrowserRouter>
      <ThemeProvider defaultTheme="system">
        <QueryClientProvider client={queryClient}>
          <App />
          {import.meta.env.DEV && <ReactQueryDevtools buttonPosition="bottom-right" />}
        </QueryClientProvider>
      </ThemeProvider>
    </BrowserRouter>
  </StrictMode>,
);