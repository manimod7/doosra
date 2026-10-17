import type { Metadata, Viewport } from "next";
import { Suspense } from "react";
import Nav, { NavFallback } from "@/components/Nav";
import { THEME_BOOT } from "@/components/ThemeToggle";
// Self-hosted variable fonts: no request to Google, same rendering everywhere.
import "@fontsource-variable/hanken-grotesk";
import "@fontsource-variable/archivo/wdth.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Doosra",
  description: "Read the match before it's bowled: forecasts, fair prices and a model that learns from every result.",
};

export const viewport: Viewport = {
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f3f5f2" },
    { media: "(prefers-color-scheme: dark)", color: "#090c0a" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body>
        <div className="shell">
          <Suspense fallback={<NavFallback />}>
            <Nav />
          </Suspense>
          <div className="wrap">{children}</div>
        </div>
      </body>
    </html>
  );
}
