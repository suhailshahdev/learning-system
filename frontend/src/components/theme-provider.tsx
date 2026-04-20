import { useEffect, useState } from "react";

import { ThemeProviderContext, type Theme } from "@/components/theme-context";

type ThemeProviderProps = {
  children: React.ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
};

export function ThemeProvider({
  children,
  defaultTheme = "system",
  storageKey = "learning-system-theme",
}: ThemeProviderProps): React.JSX.Element {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(storageKey) as Theme | null) ?? defaultTheme,
  );

  useEffect(() => {
    const root = window.document.documentElement;
    const systemQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = (): void => {
      root.classList.remove("light", "dark");
      if (theme === "system") {
        root.classList.add(systemQuery.matches ? "dark" : "light");
      } else {
        root.classList.add(theme);
      }
    };

    applyTheme();

    // Only listen for OS changes while the user is on "system".
    if (theme === "system") {
      systemQuery.addEventListener("change", applyTheme);
      return () => {
        systemQuery.removeEventListener("change", applyTheme);
      };
    }
    return undefined;
  }, [theme]);

  const value = {
    theme,
    setTheme: (newTheme: Theme) => {
      localStorage.setItem(storageKey, newTheme);
      setTheme(newTheme);
    },
  };

  return (
    <ThemeProviderContext.Provider value={value}>
      {children}
    </ThemeProviderContext.Provider>
  );
}