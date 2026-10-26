import type { ReactNode } from "react";
import { label } from "@/lib/api";

export type TraceStep = {
  type: "reasoning" | "message" | "tool_call" | "tool_result" | "privacy" | "final" | "limit" | "error" | "note";
  turn?: number;
  name?: string;
  arguments?: string;
  result?: string;
  text?: string;
  prompt?: string;
  objects?: number;
  regions?: number;
  summary?: string;
  findings?: { violation_type: string; confidence: number }[];
};

type SegObject = { object_id: string; frames: string; frame_count: number; mean_confidence: number | null };
type Submitted = { violation_type: string; sam_object: string };

type Item = { kind: "thought" | "search" | "rules" | "submit" | "note" | "warning"; turn?: number; body: ReactNode };

function parse<T>(text?: string): T | null {
  try {
    return text ? (JSON.parse(text) as T) : null;
  } catch {
    return null; // long results are stored truncated
  }
}

const pct = (n: number | null) => (n == null ? "n/a" : `${Math.round(n * 100)}%`);

/** Turns raw trace steps into plain-language timeline items. */
function toItems(steps: TraceStep[]) {
  const items: Item[] = [];
  const emptyTurns: number[] = []; // turns of searches that found nothing and await a retry
  let searches = 0;
  let retries = 0;
  let rejected = 0;

  for (let i = 0; i < steps.length; i++) {
    const s = steps[i];
    const next = steps[i + 1];
    const result = s.type === "tool_call" && next?.type === "tool_result" && next.name === s.name ? next : undefined;
    if (result) i++;

    if (s.type === "reasoning" || s.type === "message") {
      items.push({ kind: "thought", turn: s.turn, body: <p className="italic text-zinc-700">“{s.text}”</p> });
    } else if (s.type === "tool_call" && s.name === "get_grap_rules") {
      const rules = parse<unknown[]>(result?.result);
      items.push({
        kind: "rules",
        turn: s.turn,
        body: <p>Looked up the GRAP rules{rules ? ` (${rules.length} rules)` : ""} to see which violation types it can report.</p>,
      });
    } else if (s.type === "tool_call" && s.name === "segment") {
      const args = parse<{ prompt: string }>(s.arguments);
      const out = parse<{ objects?: SegObject[]; error?: string; note?: string }>(result?.result);
      const objects = out?.objects ?? [];
      // A retry is a search in a later turn than one that came back empty.
      const isRetry = emptyTurns.length > 0 && emptyTurns[0] < (s.turn ?? 0);
      if (isRetry) {
        emptyTurns.shift();
        retries++;
      }
      if (!out?.error && !out?.note?.startsWith("Already")) searches++;
      if (!out?.error && objects.length === 0) emptyTurns.push(s.turn ?? 0);
      items.push({
        kind: "search",
        turn: s.turn,
        body: (
          <div>
            <p>
              {isRetry && <span className="mr-2 rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800">Retry</span>}
              Asked SAM 3.1 to find <strong>“{args?.prompt ?? "?"}”</strong>
            </p>
            {out?.error ? (
              <p className="mt-1 text-amber-700">{out.error}</p>
            ) : objects.length === 0 ? (
              <p className="mt-1 text-zinc-600">Nothing found. The agent will try different wording.</p>
            ) : (
              <ul className="mt-1 space-y-0.5 text-zinc-600">
                {objects.map((o) => (
                  <li key={o.object_id}>
                    Found object {o.object_id}: seen in {o.frame_count} frame{o.frame_count === 1 ? "" : "s"} ({o.frames}),
                    average confidence {pct(o.mean_confidence)}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ),
      });
    } else if (s.type === "tool_call" && s.name === "submit_findings") {
      const args = parse<{ findings: Submitted[] }>(s.arguments);
      const out = parse<{ status?: string; accepted?: number; rejected?: { index: number; reason: string }[]; error?: string }>(
        result?.result,
      );
      const n = args?.findings.length;
      if (out?.status === "rejected") {
        rejected += out.rejected?.length ?? 0;
        items.push({
          kind: "warning",
          turn: s.turn,
          body: (
            <div>
              <p>
                Submitted {n ?? "its"} finding{n === 1 ? "" : "s"}. The checker <strong>rejected {out.rejected?.length}</strong>{" "}
                and sent {out.rejected?.length === 1 ? "it" : "them"} back:
              </p>
              <ul className="mt-1 list-disc pl-5 text-zinc-700">
                {out.rejected?.map((r) => {
                  const f = args?.findings[r.index];
                  return (
                    <li key={r.index}>
                      {f ? `${label(f.violation_type)}: ` : ""}
                      {r.reason.includes("never returned")
                        ? "it pointed at an object SAM never found, so there is no evidence for it."
                        : r.reason}
                    </li>
                  );
                })}
              </ul>
            </div>
          ),
        });
      } else if (out?.status === "accepted") {
        items.push({
          kind: "submit",
          turn: s.turn,
          body: <p>Submitted {out.accepted} finding{out.accepted === 1 ? "" : "s"}, each backed by a SAM object. Accepted.</p>,
        });
      } else {
        items.push({ kind: "warning", turn: s.turn, body: <p>Submission was not valid: {out?.error ?? "unreadable"}</p> });
      }
    } else if (s.type === "privacy") {
      const what = s.prompt === "face" ? "faces" : "number plates";
      items.push({
        kind: "note",
        body: (
          <p>
            Privacy check: searched for {what} ({s.objects ? `${s.objects} found, blurred before saving` : "none found"}).
          </p>
        ),
      });
    } else if (s.type === "limit" || s.type === "error") {
      items.push({ kind: "warning", turn: s.turn, body: <p>{s.text}</p> });
    } else if (s.type === "note") {
      items.push({ kind: "note", body: <p>{s.text}</p> });
    }
  }
  const turns = Math.max(0, ...steps.map((s) => s.turn ?? 0));
  return { items, stats: { turns, searches, retries, rejected } };
}

const DOT: Record<Item["kind"], string> = {
  thought: "bg-zinc-400",
  rules: "bg-blue-500",
  search: "bg-blue-500",
  submit: "bg-green-600",
  note: "bg-zinc-400",
  warning: "bg-amber-500",
};

export default function AgentTrace({ steps }: { steps: TraceStep[] }) {
  const { items, stats } = toItems(steps);
  const final = steps.find((s) => s.type === "final");

  return (
    <details className="group mt-6 rounded-lg border border-zinc-200 bg-white">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-4">
        <div>
          <h2 className="font-medium">How the agent decided</h2>
          <p className="mt-0.5 text-sm text-zinc-600">
            {stats.turns} model turns · {stats.searches} SAM searches · {stats.retries} retr{stats.retries === 1 ? "y" : "ies"} ·{" "}
            {stats.rejected} finding{stats.rejected === 1 ? "" : "s"} rejected
          </p>
        </div>
        <span className="text-sm text-zinc-500 group-open:hidden">Show steps ▾</span>
        <span className="hidden text-sm text-zinc-500 group-open:inline">Hide ▴</span>
      </summary>

      <div className="border-t border-zinc-200 px-4 pb-4">
        <p className="mt-3 text-sm text-zinc-600">
          Muse Spark looked at the media, guessed what might be a violation, and then had to prove each guess with SAM 3.1
          segmentation. A checker only accepts findings that point at an object SAM actually found.
        </p>
        <ol className="mt-4 space-y-3 border-l border-zinc-200 pl-5 text-sm">
          {items.map((item, i) => {
            const header = item.turn !== undefined && item.turn !== items[i - 1]?.turn;
            return (
              <li key={i} className="relative">
                {header && <p className="-ml-5 mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Turn {item.turn}</p>}
                <span className={`absolute -left-[25px] mt-1.5 h-2.5 w-2.5 rounded-full ${DOT[item.kind]}`} />
                {item.body}
              </li>
            );
          })}
          {final && (
            <li className="relative">
              <span className="absolute -left-[25px] mt-1.5 h-2.5 w-2.5 rounded-full bg-green-600" />
              <p className="font-medium">Result</p>
              <p className="mt-1 text-zinc-700">{final.summary}</p>
            </li>
          )}
        </ol>
      </div>
    </details>
  );
}
