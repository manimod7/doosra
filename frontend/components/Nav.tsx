"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "@/components/ThemeToggle";

const ITEMS = [
  { href: "/", label: "Today", icon: "M4 11.5 12 5l8 6.5V19a1 1 0 0 1-1 1h-4.5v-5.5h-5V20H5a1 1 0 0 1-1-1z" },
  { href: "/forecast", label: "Forecast", icon: "M4 19c3-1 4.5-7 8-7s4 3 8-7M4 19h16" },
  { href: "/ledger", label: "Ledger", icon: "M6 4h12v16H6zM9 8h6M9 12h6M9 16h3" },
  { href: "/learning", label: "Learning", icon: "M12 4a8 8 0 1 0 8 8M12 4v8l5-5" },
];

export function Mark() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="var(--ball)" />
      <path d="M8.5 6.8c4.6 4.8 4.6 13.6 0 18.4M23.5 6.8c-4.6 4.8-4.6 13.6 0 18.4" fill="none"
        stroke="#fff" strokeWidth="1.6" strokeLinecap="round" strokeDasharray="0.1 3.1" />
    </svg>
  );
}

/** Shown while the current route resolves: same bar, no active tab yet. */
export function NavFallback() {
  return (
    <header className="bar">
      <div className="bar-in">
        <Link href="/" className="brand" aria-label="Doosra, home"><Mark /><span className="brand-word">Doosra</span></Link>
        <div className="bar-end">
          <nav className="nav" aria-label="Sections">
            {ITEMS.map((i) => <Link key={i.href} href={i.href}>{i.label}</Link>)}
          </nav>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

export default function Nav() {
  const path = usePathname();
  const current = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  return (
    <>
      <header className="bar">
        <div className="bar-in">
          <Link href="/" className="brand" aria-label="Doosra, home">
            <Mark />
            <span className="brand-word">Doosra</span>
          </Link>
          <div className="bar-end">
            <nav className="nav" aria-label="Sections">
              {ITEMS.map((i) => (
                <Link key={i.href} href={i.href} aria-current={current(i.href) ? "page" : undefined}>
                  {i.label}
                </Link>
              ))}
            </nav>
            <ThemeToggle />
          </div>
        </div>
      </header>
      <nav className="tabbar" aria-label="Sections">
        {ITEMS.map((i) => (
          <Link key={i.href} href={i.href} aria-current={current(i.href) ? "page" : undefined}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
              strokeLinejoin="round" aria-hidden="true">
              <path d={i.icon} />
            </svg>
            {i.label}
          </Link>
        ))}
      </nav>
    </>
  );
}
