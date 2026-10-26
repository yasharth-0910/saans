"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";
import MapView from "@/components/Map";
import { API_URL } from "@/lib/api";

export default function NewReport() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [picked, setPicked] = useState<{ lat: number; lng: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file || !picked) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    body.append("lat", String(picked.lat));
    body.append("lng", String(picked.lng));
    try {
      const res = await fetch(`${API_URL}/reports`, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `Upload failed (${res.status})`);
      router.push(`/reports/${data.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8">
      <h1 className="text-2xl font-semibold">Report a violation</h1>
      <p className="mt-1 text-zinc-600">
        Upload a photo or short video, then click the map where it was taken.
      </p>

      <form onSubmit={submit} className="mt-6 space-y-6">
        <label className="block">
          <span className="text-sm font-medium">Photo or video</span>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-2 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-zinc-200 file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-zinc-300"
          />
        </label>

        <div>
          <span className="text-sm font-medium">Location</span>
          <div className="mt-2 h-80 overflow-hidden rounded-lg border border-zinc-200">
            <MapView picked={picked} onPick={(lat, lng) => setPicked({ lat, lng })} />
          </div>
          <p className="mt-2 text-sm text-zinc-600">
            {picked ? `${picked.lat.toFixed(5)}, ${picked.lng.toFixed(5)}` : "Click the map to drop a pin."}
          </p>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button
          type="submit"
          disabled={!file || !picked || busy}
          className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-400"
        >
          {busy ? "Analysing…" : "Submit report"}
        </button>
      </form>
    </main>
  );
}
