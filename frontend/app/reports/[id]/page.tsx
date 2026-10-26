import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import AgentTrace, { type TraceStep } from "@/components/AgentTrace";
import EvidenceVideo from "@/components/EvidenceVideo";
import { API_URL, type Report, label, stageName } from "@/lib/api";

type Params = { params: Promise<{ id: string }> };

export default function ReportPage({ params }: Params) {
  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-8">
      <Link href="/" className="text-sm text-zinc-600 hover:underline">
        ← Back to map
      </Link>
      <h1 className="mt-3 text-2xl font-semibold">Evidence report</h1>
      {/* Report data is fetched per request, so it streams in behind this fallback. */}
      <Suspense fallback={<p className="mt-2 text-sm text-zinc-500">Loading report…</p>}>
        <ReportBody params={params} />
      </Suspense>
    </main>
  );
}

async function ReportBody({ params }: Params) {
  const { id } = await params;
  const res = await fetch(`${API_URL}/reports/${encodeURIComponent(id)}`, { cache: "no-store" });
  if (res.status === 404) notFound();
  if (!res.ok) throw new Error(`API error ${res.status}`);
  const report: Report = await res.json();
  const traceRes = await fetch(`${API_URL}/reports/${encodeURIComponent(id)}/trace`, { cache: "no-store" });
  const steps: TraceStep[] = traceRes.ok ? (await traceRes.json()).steps : [];

  return (
    <>
      <p className="mt-1 text-sm text-zinc-600">
        {new Date(report.created_at).toLocaleString()} · {report.lat.toFixed(5)}, {report.lng.toFixed(5)} ·{" "}
        {report.media_type}
      </p>

      <section className="mt-6 rounded-lg border border-zinc-200 bg-white p-4">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">Summary</h2>
        <p className="mt-2">{report.summary}</p>
      </section>

      {steps.length > 0 && <AgentTrace steps={steps} />}

      {report.media_type === "video" && (
        <section className="mt-6">
          <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">Video evidence</h2>
          <EvidenceVideo reportId={report.id} labels={report.findings.map((f) => label(f.violation_type))} />
        </section>
      )}

      <section className="mt-6">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
          {report.media_type === "video" ? "Key frames" : "Evidence frames"}
        </h2>
        {report.frames.length ? (
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            {report.frames.map((src) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={src} src={`${API_URL}${src}`} alt="Evidence frame" className="w-full rounded-lg border border-zinc-200" />
            ))}
          </div>
        ) : (
          <p className="mt-2 text-sm text-zinc-600">
            {report.media_type === "seed" ? "Demo report seeded without media." : "No evidence frames."}
          </p>
        )}
      </section>

      <section className="mt-6">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
          Findings ({report.findings.length})
        </h2>
        <ul className="mt-3 space-y-3">
          {report.findings.map((f, i) => (
            <li key={i} className="rounded-lg border border-zinc-200 bg-white p-4">
              <div className="flex items-baseline justify-between gap-4">
                <h3 className="font-medium">{label(f.violation_type)}</h3>
                <span className="text-sm text-zinc-600">
                  {Math.round(f.confidence * 100)}% confidence
                  {report.media_type === "video" && ` · frame ${f.frame_index}`}
                </span>
              </div>
              <div className="mt-2 h-1.5 rounded-full bg-zinc-100">
                <div className="h-1.5 rounded-full bg-zinc-800" style={{ width: `${f.confidence * 100}%` }} />
              </div>
              <p className="mt-3 text-sm">{f.description}</p>
              {f.rule ? (
                <div className="mt-3 rounded-md bg-zinc-50 p-3 text-sm">
                  <p className="font-medium">{f.rule.title}</p>
                  <p className="text-zinc-500">Applies from GRAP Stage {stageName(f.rule.stage)}</p>
                  <p className="mt-1 text-zinc-700">{f.rule.rule_text}</p>
                </div>
              ) : (
                <p className="mt-3 text-sm text-zinc-500">Rule {f.rule_id} not found.</p>
              )}
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
