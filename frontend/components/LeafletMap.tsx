"use client";

// Imports Leaflet at module level, so load this only with next/dynamic and ssr: false.
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect, useRef } from "react";
import { type Hotspot, type Report, label } from "@/lib/api";

const NCR_CENTER: L.LatLngTuple = [28.58, 77.4];

type Props = {
  reports?: Report[];
  hotspots?: Hotspot[];
  picked?: { lat: number; lng: number } | null;
  onPick?: (lat: number, lng: number) => void;
};

function popup(report: Report) {
  const div = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = report.findings.map((f) => label(f.violation_type)).join(", ") || "No findings";
  const date = document.createElement("div");
  date.textContent = new Date(report.created_at).toLocaleString();
  const link = document.createElement("a");
  link.href = `/reports/${report.id}`;
  link.textContent = "Open report →";
  div.append(title, date, link);
  return div;
}

export default function LeafletMap({ reports = [], hotspots = [], picked, onPick }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const layers = useRef<L.LayerGroup | null>(null);
  const pickRef = useRef(onPick);

  useEffect(() => {
    pickRef.current = onPick;
  }, [onPick]);

  useEffect(() => {
    const m = L.map(container.current!).setView(NCR_CENTER, 11);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(m);
    m.on("click", (e) => pickRef.current?.(e.latlng.lat, e.latlng.lng));
    map.current = m;
    layers.current = L.layerGroup().addTo(m);
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    const group = layers.current;
    if (!group) return;
    group.clearLayers();
    const inHotspot = new Set(hotspots.flatMap((h) => h.report_ids));

    for (const h of hotspots) {
      // Pixel radius, so hotspots stay visible at any zoom.
      L.circleMarker([h.lat, h.lng], {
        radius: 16 + 6 * h.count,
        color: "#dc2626",
        fillColor: "#ef4444",
        fillOpacity: 0.25,
        weight: 2,
      })
        .bindTooltip(`Hotspot: ${h.count} reports within 200 m`)
        .addTo(group);
    }
    for (const r of reports) {
      const hot = inHotspot.has(r.id);
      L.circleMarker([r.lat, r.lng], {
        radius: hot ? 9 : 7,
        color: "#fff",
        weight: 2,
        fillColor: hot ? "#dc2626" : "#2563eb",
        fillOpacity: 1,
      })
        .bindPopup(popup(r))
        .addTo(group);
    }
    if (picked) {
      L.circleMarker([picked.lat, picked.lng], {
        radius: 10,
        color: "#fff",
        weight: 3,
        fillColor: "#16a34a",
        fillOpacity: 1,
      }).addTo(group);
    }
  }, [reports, hotspots, picked]);

  return <div ref={container} className="h-full w-full" />;
}
