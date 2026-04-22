import { useContext } from "react";

import { ThemeProviderContext, type ThemeProviderState } from "@/components/theme-context";

export function useTheme(): ThemeProviderState {
  const context = useContext(ThemeProviderContext);
  if (context === null) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
