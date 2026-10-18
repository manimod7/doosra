# Deploying Doosra (free tiers)

Keys, passwords and connection strings go only into the places below, never into code.
`scripts/check-secrets.sh` blocks commits that contain them, locally and in CI.

| Piece | Where | Settings |
|---|---|---|
| Database | Neon (project `doosra`, Singapore, Postgres only) | Copy the connection string with **Connection pooling off** |
| API | Google Cloud Run, service `doosra-api`, region `asia-south1` | Variables: `DOOSRA_DB`, `DOOSRA_ADMIN_TOKEN`, `DOOSRA_ODDS_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `DOOSRA_CORS` |
| Dashboard | Vercel, root directory `frontend` | Variables: `DOOSRA_API_URL`, `DOOSRA_ADMIN_TOKEN`, `DOOSRA_DASHBOARD_PASSWORD` |
| Weekly learning, odds refresh, smoke test | GitHub Actions | Secrets: `DOOSRA_DB`, `DOOSRA_API_URL`, `DOOSRA_ADMIN_TOKEN`; optional `DOOSRA_SITE_URL`, `DOOSRA_DASHBOARD_PASSWORD` |

## 1. Google Cloud Run (API)

1. Project `doosra`: **Billing → Link a billing account**, then **Budgets & alerts** (₹100, email).
2. Enable the APIs: Cloud Run, Cloud Build, Artifact Registry, Developer Connect, Container Analysis, IAM
   (`https://console.cloud.google.com/flows/enableapi?apiid=run.googleapis.com,cloudbuild.googleapis.com,artifactregistry.googleapis.com,developerconnect.googleapis.com,containeranalysis.googleapis.com,iam.googleapis.com&project=<your-project-id>`).
3. **Cloud Run → Deploy container → Service → Continuously deploy from a repository → Cloud Build**:
   GitHub `manimod7/doosra`, branch `^main$`, build type **Dockerfile**, source location **`/Dockerfile`**.
4. Service name `doosra-api`, region `asia-south1`, **Allow public access**, **Request-based** billing,
   instances min 0 / max 1, ingress **All**.
5. **Containers → Resources**: 2 GiB memory, 1 CPU, port 8080. **Variables & secrets**: the API row above
   (leave `DOOSRA_CORS` until step 3 below).
6. **Create**. The first build takes 5 to 8 minutes. Copy the service URL.

The admin token protects everything that writes or spends Odds API credits; reads (forecasts, prices) are open.

## 2. Vercel (dashboard)

**Add New → Project**, import `manimod7/doosra`, **Root Directory** `frontend`, add the dashboard variables, **Deploy**.
`DOOSRA_API_URL` is the Cloud Run URL; `DOOSRA_ADMIN_TOKEN` must equal the API's.

## 3. Connect them

1. Cloud Run → `doosra-api` → **Edit & deploy new revision** → add `DOOSRA_CORS` = your Vercel address → **Deploy**.
2. GitHub → **Settings → Secrets and variables → Actions**: add the secrets in the table.

## 4. First model and checks

1. **Actions → Learning cycle → Run workflow** (about 2 to 3 hours on GitHub's free runner). It downloads
   Cricsheet, trains, and stores the winning model in Neon.
2. Nothing to restart: the API checks the database for a newer model at most every 10 minutes and switches
   to it on its own (and always loads the newest one when it starts).
3. **Actions → Smoke test → Run workflow**. Every line should say PASS.

After that, the learning cycle runs every Monday (03:10 IST) and odds refresh twice a day.

## What each piece costs

Everything above sits inside the free tiers for one person's use: Cloud Run scales to zero between requests,
Neon sleeps when idle, Vercel Hobby and GitHub Actions free minutes cover the dashboard and the weekly jobs.
The budget alert is the safety net.
