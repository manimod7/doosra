import { NextResponse, type NextRequest } from "next/server";

// Password gate for the whole dashboard (HTTP Basic auth, any username).
// Locally with no password set it stays open; in production it fails closed.
export function proxy(request: NextRequest) {
  const password = process.env.DOOSRA_DASHBOARD_PASSWORD;
  if (!password) {
    if (process.env.NODE_ENV === "production") {
      return new NextResponse(
        "Doosra is locked: set DOOSRA_DASHBOARD_PASSWORD on the host.",
        { status: 503 },
      );
    }
    return NextResponse.next();
  }
  const header = request.headers.get("authorization") ?? "";
  if (header.startsWith("Basic ")) {
    try {
      const decoded = atob(header.slice(6));
      const given = decoded.slice(decoded.indexOf(":") + 1);
      if (safeEqual(given, password)) return NextResponse.next();
    } catch {
      // fall through to the challenge
    }
  }
  return new NextResponse("Password required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Doosra", charset="UTF-8"' },
  });
}

function safeEqual(a: string, b: string) {
  let diff = a.length ^ b.length;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }
  return diff === 0;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
