import { NextResponse, type NextRequest } from "next/server";
import { SERVER_API } from "@/lib/api";

// Server-side bridge to the Doosra API. The browser never sees the API URL
// or the admin token; only the paths listed here can be reached.
const ALLOWED: Record<"GET" | "POST", RegExp[]> = {
  GET: [/^matches\/[\w.-]+\/odds$/, /^budget$/, /^bets$/, /^engine$/, /^catalog$/],
  POST: [
    /^matches\/[\w.-]+\/explain$/,
    /^matches\/[\w.-]+\/flags$/,
    /^markets\/card$/,
    /^markets\/price$/,
    /^admin\/refresh$/,
    /^bets$/,
    /^bets\/\d+\/settle$/,
  ],
};

async function forward(method: "GET" | "POST", request: NextRequest, params: Promise<{ path: string[] }>) {
  const path = (await params).path.join("/");
  if (!ALLOWED[method].some((re) => re.test(path))) {
    return NextResponse.json({ detail: "Not available" }, { status: 404 });
  }
  const url = new URL(`${SERVER_API}/${path}`);
  request.nextUrl.searchParams.forEach((v, k) => url.searchParams.set(k, v));
  const headers: Record<string, string> = { "content-type": "application/json" };
  const token = process.env.DOOSRA_ADMIN_TOKEN;
  if (token) headers.authorization = `Bearer ${token}`;
  try {
    const res = await fetch(url, {
      method, headers, cache: "no-store",
      body: method === "POST" ? await request.text() : undefined,
    });
    return new NextResponse(await res.text(), {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json({ detail: "Doosra's engine isn't reachable right now." }, { status: 502 });
  }
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  return forward("GET", request, params);
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  return forward("POST", request, params);
}
