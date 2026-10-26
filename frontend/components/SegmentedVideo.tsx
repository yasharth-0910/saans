"use client";

// Imports @meta-sam/react at module level, so load this only with next/dynamic and ssr: false.
import { objectColor, SegmentationRenderer } from "@meta-sam/graphics";
import { parseVideoStream, type VideoSegmentationResult } from "@meta-sam/parser";
import { Video, type VideoRef } from "@meta-sam/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { API_URL } from "@/lib/api";

type Overlay = {
  video: string;
  frame_width: number;
  frame_height: number;
  tracks: { track_id: string; finding: number; prompt: string }[];
  wire: string;
};

// Replays the saved SAM wire lines as a Responses stream so @meta-sam/parser builds the result.
async function* wireEvents(wire: string) {
  const lane = { item_id: "overlay", output_index: 0, content_index: 0 };
  for (const line of wire.split("\n")) {
    if (line) yield { type: "response.output_text.delta" as const, delta: `${line}\n`, ...lane };
  }
  yield { type: "response.output_text.done" as const, text: wire, ...lane };
  yield { type: "response.completed" as const };
}

export default function SegmentedVideo({ reportId, labels }: { reportId: string; labels: string[] }) {
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [result, setResult] = useState<VideoSegmentationResult | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const videoRef = useRef<VideoRef>(null);
  // Not disposed in an effect cleanup: StrictMode's dev remount would leave Video holding a dead
  // renderer. It holds only this report's overlay and is garbage-collected with the component.
  const renderer = useMemo(() => new SegmentationRenderer({ boxLabels: true }), []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const res = await fetch(`${API_URL}/reports/${reportId}/overlay`);
      if (!res.ok) throw new Error(`Could not load the video overlay (${res.status})`);
      const data: Overlay = await res.json();
      const parsed = await parseVideoStream(wireEvents(data.wire)).finalResult;
      if (!cancelled) {
        setOverlay(data);
        setResult(parsed);
      }
    })().catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [reportId]);

  function toggle(trackId: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
  }

  if (error) return <p className="mt-2 text-sm text-red-600">{error}</p>;
  if (!overlay || !result) return <div className="mt-3 aspect-video w-full animate-pulse rounded-lg bg-zinc-100" />;

  return (
    <div className="mt-3 space-y-3">
      <div className="overflow-hidden rounded-lg border border-zinc-200 bg-black">
        <Video
          ref={videoRef}
          src={`${API_URL}${overlay.video}`}
          result={result}
          renderer={renderer}
          hiddenIds={hidden}
          boxLabel="Finding"
          objectFit="contain"
          loop
          muted
          onPlayingChange={setPlaying}
          onError={(e) => setError(`Video playback failed: ${e.message}`)}
          style={{ width: "100%", aspectRatio: `${overlay.frame_width} / ${overlay.frame_height}`, display: "block" }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => (playing ? videoRef.current?.pause() : void videoRef.current?.play())}
          className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-zinc-700"
        >
          {playing ? "Pause" : "Play"}
        </button>
        {overlay.tracks.map((t) => (
          <label
            key={t.track_id}
            className="flex cursor-pointer items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-sm"
          >
            <input type="checkbox" checked={!hidden.has(t.track_id)} onChange={() => toggle(t.track_id)} />
            <span className="h-3 w-3 rounded-sm" style={{ background: objectColor(t.track_id) }} />
            Finding {t.track_id}: {labels[t.finding] ?? "object"}
            <span className="text-zinc-500">(SAM: “{t.prompt}”)</span>
          </label>
        ))}
      </div>
      <p className="text-xs text-zinc-500">
        Faces and number plates are blurred on every frame before the video is stored. Overlays are SAM 3.1 masks
        tracked across frames.
      </p>
    </div>
  );
}
