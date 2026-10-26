# Deploying Saans

Backend: Google Cloud Run. Frontend: Vercel. Do the backend first, because the frontend needs its URL.

Run every command from the repo root.

## 1. Backend on Google Cloud Run

### Install the gcloud CLI (macOS)

```bash
brew install --cask gcloud-cli
```

```bash
gcloud init
```

`gcloud init` opens the browser to sign in. When it asks for a project, choose **Create a new project** or skip; the next step creates one.

### Create a project

The project ID must be unique across all of Google Cloud: lowercase letters, digits and hyphens, 6 to 30 characters. Pick one and keep it in a shell variable for the rest of this section:

```bash
PROJECT_ID=saans-demo-<something-unique>
```

```bash
gcloud projects create $PROJECT_ID --name="Saans"
```

```bash
gcloud config set project $PROJECT_ID
```

### Enable billing

Cloud Run needs a billing account, even though this setup should stay within the free tier. If you don't have one, create it at https://console.cloud.google.com/billing (card required). Then find its ID:

```bash
gcloud billing accounts list
```

```bash
gcloud billing projects link $PROJECT_ID --billing-account=<ACCOUNT_ID>
```

### Set a budget alert

In the console: **Billing → Budgets & alerts → Create budget**.

1. Scope: this project only.
2. Amount: a small fixed amount, for example 5 USD or 500 INR (it uses your billing account's currency).
3. Alert thresholds: 50%, 90% and 100% of actual spend. Leave **Email alerts to billing admins** on.
4. Click **Finish**.

A budget only sends emails. It does not stop spending. `max-instances 1` below is what keeps the cost bounded.

### Enable the APIs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

### Store the API key in Secret Manager

This reads the key from your local `.env`, so it never appears on screen or in your shell history:

```bash
grep '^MODEL_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | gcloud secrets create model-api-key --data-file=-
```

Let Cloud Run read it, and let Cloud Build build from source. Both run as the project's default compute service account:

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
```

```bash
gcloud secrets add-iam-policy-binding model-api-key --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" --role=roles/secretmanager.secretAccessor
```

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" --role=roles/run.builder
```

### Deploy

```bash
gcloud run deploy saans-api --source . --region asia-south1 --memory 2Gi --cpu 1 --min-instances 0 --max-instances 1 --timeout 900 --allow-unauthenticated --set-env-vars "^@^USE_MOCK=1@FRONTEND_ORIGIN=http://localhost:3000" --set-secrets MODEL_API_KEY=model-api-key:latest
```

- The first time, it asks to create an Artifact Registry repository for the image. Answer **Y**.
- `--source .` uploads the repo, minus everything in `.gcloudignore` (all `.env` files, the frontend, the local database and uploads), and builds it with the `Dockerfile` on Cloud Build.
- `--max-instances 1` keeps a single copy of the SQLite database, so reports stay consistent. `--min-instances 0` lets the service scale to zero when idle, so it costs nothing then.
- `--timeout 900` gives a real (`USE_MOCK=0`) video analysis up to 15 minutes. The default is 5.
- `^@^` at the start of `--set-env-vars` makes `@` the separator between variables, because `FRONTEND_ORIGIN` itself contains commas.
- `FRONTEND_ORIGIN` is a placeholder for now; step 3 sets it to the Vercel URL.

When it finishes it prints the **Service URL**, for example `https://saans-api-xxxxxxxxxx.asia-south1.run.app`. To print it again:

```bash
gcloud run services describe saans-api --region asia-south1 --format='value(status.url)'
```

Run the same `gcloud run deploy` command again after every backend change.

### Check it

```bash
curl <SERVICE_URL>/reports
```

You should get the 8 seeded demo reports. The first request after an idle spell takes a few seconds while an instance starts.

### Change env vars or the key later (no rebuild)

These create a new revision from the same image, so they take seconds:

```bash
gcloud run services update saans-api --region asia-south1 --update-env-vars "^@^USE_MOCK=0@FRONTEND_ORIGIN=https://saans.vercel.app,http://localhost:3000"
```

To replace the API key, add a new secret version (again read from `.env`), then roll a revision so the service picks it up:

```bash
grep '^MODEL_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | gcloud secrets versions add model-api-key --data-file=-
```

```bash
gcloud run services update saans-api --region asia-south1 --update-secrets MODEL_API_KEY=model-api-key:latest
```

### What to expect from storage

The container's filesystem is not persistent. It resets whenever the instance restarts, which with `min-instances 0` happens after it has been idle for a while, and on every deploy or env var change. User-submitted reports and their evidence can disappear then; only the seeded demo reports are guaranteed, because the API reseeds an empty database on startup. That is acceptable for the demo. Files written there also count against the 2 GiB of memory.

Cloud Run rejects request bodies over 32 MiB, so uploads larger than that fail even though the API itself allows 200 MB. Keep demo clips short.

## 2. Frontend on Vercel

The code has to be on GitHub first (push the repo yourself).

1. Go to https://vercel.com/new and import the GitHub repo.
2. **Root Directory**: click **Edit** and pick `frontend`. Vercel detects Next.js; leave the build settings as they are.
3. **Environment Variables**: add `NEXT_PUBLIC_API_URL` = your Cloud Run service URL (no trailing slash), for Production and Preview.
4. Click **Deploy**.

`NEXT_PUBLIC_API_URL` is baked in at build time, and the build fails if it is missing. If you change it later, redeploy: **Deployments → ⋯ on the latest → Redeploy**.

Note the production URL Vercel gives you, for example `https://saans.vercel.app`.

## 3. Connect the two

1. Allow the Vercel URL in the API's CORS list:

   ```bash
   gcloud run services update saans-api --region asia-south1 --update-env-vars "^@^FRONTEND_ORIGIN=https://saans.vercel.app,http://localhost:3000"
   ```

2. Open the Vercel URL. The map should show the demo reports. Open one, then try **/new** with a photo.

Vercel preview deployments get their own URLs, which are not in `FRONTEND_ORIGIN`, so their API calls are blocked by CORS. Add a preview URL to the list if you need it.

## Switching to the real agent

```bash
gcloud run services update saans-api --region asia-south1 --update-env-vars USE_MOCK=0
```

`MODEL_API_KEY` is already wired in from Secret Manager. No frontend change needed.

## Testing the image locally

```bash
docker build -t saans-api . && docker run --rm -p 8080:8080 -e FRONTEND_ORIGIN=http://localhost:3000 saans-api
```

```bash
curl 127.0.0.1:8080/reports
```
