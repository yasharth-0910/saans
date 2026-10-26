# Deploying Saans

Backend: Hugging Face Spaces (Docker). Frontend: Vercel. Do the backend first, because the frontend needs its URL.

## 1. Backend on Hugging Face Spaces

### Create the Space

1. Go to https://huggingface.co/new-space.
2. Name it, for example `saans-api`.
3. SDK: **Docker**, template: **Blank**.
4. Hardware: **CPU basic** (free).
5. Visibility: **Public**. The browser calls the API directly, and a private Space would need a token.
6. Click **Create Space**. Keep the README.md it creates: its header (`sdk: docker`) tells Spaces to build the Dockerfile, and the default port is 7860, which is what the app listens on.

The API URL will be `https://<your-username>-saans-api.hf.space` (lowercase, dots and underscores become hyphens). You can copy it from the Space page: **⋯ menu → Embed this Space → Direct URL**.

### Set secrets and variables

In the Space: **Settings → Variables and secrets**.

| Kind | Name | Value |
|---|---|---|
| Secret | `MODEL_API_KEY` | your Meta Model API key |
| Variable | `USE_MOCK` | `1` for the scripted demo agent, `0` for real Meta API calls |
| Variable | `FRONTEND_ORIGIN` | `http://localhost:3000` for now; replaced with the Vercel URL in step 3 |

`FRONTEND_ORIGIN` is a comma-separated list, for example `https://saans.vercel.app,http://localhost:3000`. No trailing slash.

### Upload the code

From the repo root, once:

```bash
backend/.venv/bin/pip install -U huggingface_hub
```

```bash
backend/.venv/bin/hf auth login
```

Paste a token with **write** access from https://huggingface.co/settings/tokens.

Then upload (run this again after every backend change; each upload triggers a rebuild):

```bash
backend/.venv/bin/hf upload <your-username>/saans-api . . --repo-type space --include "Dockerfile" --include "backend/*" --include "data/grap_rules.json" --exclude "*/.venv/*" --exclude "*__pycache__*" --exclude "*.pytest_cache*"
```

This sends only the Dockerfile, `backend/` and the rules file. Your `.env`, database and uploads stay local.

### Check it

Watch the **Logs** tab until it shows `Uvicorn running on http://0.0.0.0:7860`. Then:

```bash
curl https://<your-username>-saans-api.hf.space/reports
```

You should get the 8 seeded demo reports. Storage on a Space is wiped whenever it restarts or rebuilds, so uploaded reports disappear and the demo reports come back automatically on the next start.

A free Space goes to sleep after 48 hours without traffic. Open the Space page a few minutes before a demo to wake it.

## 2. Frontend on Vercel

The code has to be on GitHub first (push the repo yourself).

1. Go to https://vercel.com/new and import the GitHub repo.
2. **Root Directory**: click **Edit** and pick `frontend`. Vercel detects Next.js; leave the build settings as they are.
3. **Environment Variables**: add `NEXT_PUBLIC_API_URL` = `https://<your-username>-saans-api.hf.space` (no trailing slash), for Production and Preview.
4. Click **Deploy**.

`NEXT_PUBLIC_API_URL` is baked in at build time, and the build fails if it is missing. If you change it later, redeploy: **Deployments → ⋯ on the latest → Redeploy**.

Note the production URL Vercel gives you, for example `https://saans.vercel.app`.

## 3. Connect the two

1. In the Space: **Settings → Variables and secrets**, set `FRONTEND_ORIGIN` to the Vercel production URL, for example `https://saans.vercel.app,http://localhost:3000`.
2. The Space restarts on its own after the change. If it doesn't, use **Settings → Factory rebuild**.
3. Open the Vercel URL. The map should show the demo reports. Open one, then try **/new** with a photo.

Vercel preview deployments get their own URLs, which are not in `FRONTEND_ORIGIN`, so their API calls are blocked by CORS. Add a preview URL to the list if you need it.

## Switching to the real agent

Set the Space variable `USE_MOCK` to `0`. `MODEL_API_KEY` must be set as a secret. No frontend change needed.
