import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Activity, Menu, X } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

const NAV_LINKS = [
  { label: "Home", href: "#home" },
  { label: "Platform", href: "#platform" },
  { label: "AI Models", href: "#ai-models" },
  { label: "How It Works", href: "#how-it-works" },
  { label: "Technology", href: "#technology" },
];

function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2.5 shrink-0">
      <span className="relative flex h-9 w-9 items-center justify-center rounded-lg bg-blue text-white">
        <Activity className="h-5 w-5" strokeWidth={2.25} aria-hidden="true" />
      </span>
      <span className="flex items-baseline gap-1">
        <span className="text-lg font-semibold tracking-tight text-navy font-sans">ERFlow</span>
        <span className="hidden sm:inline text-[11px] font-medium uppercase tracking-[0.14em] text-navy-soft">
          Emergency Intelligence Platform
        </span>
      </span>
    </Link>
  );
}

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <header
      className={`sticky top-0 z-50 w-full border-b bg-surface/90 backdrop-blur transition-shadow ${
        scrolled ? "border-border shadow-soft" : "border-transparent"
      }`}
    >
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Logo />

        <ul className="hidden items-center gap-8 lg:flex">
          {NAV_LINKS.map((link) => (
            <li key={link.label}>
              <a
                href={link.href}
                className="text-[15px] font-medium text-navy-muted transition-colors hover:text-blue"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="hidden lg:flex items-center gap-3">
          <ThemeToggle />
          <Link
            to="/dashboard"
            className="inline-flex items-center rounded-lg bg-blue px-4 py-2.5 text-[15px] font-semibold text-white shadow-soft transition-colors hover:bg-blue-dark"
          >
            Open Dashboard
          </Link>
        </div>

        <div className="flex items-center gap-2 lg:hidden">
          <ThemeToggle />
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label={open ? "Close menu" : "Open menu"}
            className="inline-flex items-center justify-center rounded-lg border border-border p-2 text-navy"
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </nav>

      {open && (
        <div className="border-t border-border bg-surface lg:hidden">
          <ul className="flex flex-col gap-1 px-4 py-3 sm:px-6">
            {NAV_LINKS.map((link) => (
              <li key={link.label}>
                <a
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="block rounded-lg px-3 py-2.5 text-[15px] font-medium text-navy-muted hover:bg-bg hover:text-blue"
                >
                  {link.label}
                </a>
              </li>
            ))}
            <li className="pt-2 flex flex-col gap-2">
              <ThemeToggle showLabel={true} />
              <Link
                to="/dashboard"
                onClick={() => setOpen(false)}
                className="block rounded-lg bg-blue px-3 py-2.5 text-center text-[15px] font-semibold text-white hover:bg-blue-dark"
              >
                Open Dashboard
              </Link>
            </li>
          </ul>
        </div>
      )}
    </header>
  );
}
