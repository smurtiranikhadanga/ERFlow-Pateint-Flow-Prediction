import React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "../context/ThemeContext";

export default function ThemeToggle({ className = "", showLabel = false }) {
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
      className={`relative inline-flex items-center gap-2 rounded-lg border border-border bg-surface text-navy-muted transition-all duration-200 hover:bg-bg hover:border-border-strong hover:text-navy focus:outline-none focus-visible:ring-2 focus-visible:ring-blue ${
        showLabel ? "w-full justify-between px-3 py-2 text-[14px]" : "h-9 w-9 justify-center"
      } ${className}`}
    >
      <div className="relative flex h-[18px] w-[18px] items-center justify-center">
        {/* Sun Icon */}
        <Sun
          className={`h-[18px] w-[18px] text-amber-500 transition-all duration-300 transform ${
            isDark
              ? "rotate-90 scale-0 opacity-0 absolute"
              : "rotate-0 scale-100 opacity-100"
          }`}
          strokeWidth={2.25}
          aria-hidden="true"
        />
        {/* Moon Icon */}
        <Moon
          className={`h-[18px] w-[18px] text-[#c0e6fd] transition-all duration-300 transform ${
            isDark
              ? "rotate-0 scale-100 opacity-100"
              : "-rotate-90 scale-0 opacity-0 absolute"
          }`}
          strokeWidth={2.25}
          aria-hidden="true"
        />
      </div>

      {showLabel && (
        <span className="font-medium text-navy">
          {isDark ? "Dark Theme" : "Light Theme"}
        </span>
      )}
    </button>
  );
}
