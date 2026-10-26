"use client";

import dynamic from "next/dynamic";

// Leaflet needs `window`, so it is only ever loaded in the browser.
const MapView = dynamic(() => import("./LeafletMap"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse bg-zinc-100" />,
});

export default MapView;
