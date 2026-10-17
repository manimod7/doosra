"use client";

/** Light or dark, remembered on this device. Starts from the system setting. */
export default function ThemeToggle() {
  const flip = () => {
    const root = document.documentElement;
    const now = root.dataset.theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = now === "dark" ? "light" : "dark";
    root.classList.add("tt");
    root.dataset.theme = next;
    try { localStorage.setItem("doosra-theme", next); } catch { /* private mode: still switches for this visit */ }
    window.setTimeout(() => root.classList.remove("tt"), 400);
  };
  return (
    <button type="button" className="theme-btn" onClick={flip} aria-label="Switch between light and dark" title="Light / dark">
      <svg className="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
      <svg className="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
    </button>
  );
}

/** Runs before first paint so a saved theme never flashes the wrong one. */
export const THEME_BOOT = `try{var t=localStorage.getItem("doosra-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;
