"use client";

import { useEffect, useState } from "react";
import MapView from "@/components/Map";
import { API_URL, type Hotspot, type Report } from "@/lib/api";

export default function Home() {
  const [reports, setReports] = useState<Report[]>([]);
  const [hotspots, setHotspots] = useState<Hotspot[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/reports`).then((r) => r.json()),
      fetch(`${API_URL}/hotspots`).then((r) => r.json()),
    ])
      .then(([r, h]) => {
        setReports(r);
        setHotspots(h);
      })
      .catch(() => setError(`Could not reach the API at ${API_URL}. Is the backend running?`));
  }, []);

  return (
    <main className="relative flex-1">
      <div className="absolute inset-0">
        <MapView reports={reports} hotspots={hotspots} />
      </div>
      <div className="absolute bottom-6 left-4 z-[1000] rounded-lg bg-white/95 px-4 py-3 text-sm shadow">
        <p className="font-medium">
          {reports.length} reports · {hotspots.length} hotspots
        </p>
        <ul className="mt-2 space-y-1 text-zinc-600">
          <li className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-blue-600" /> Report
          </li>
          <li className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-red-600" /> Part of a hotspot (2+ reports within 200 m)
          </li>
        </ul>
        {error && <p className="mt-2 text-red-600">{error}</p>}
      </div>
    </main>
  );
}
