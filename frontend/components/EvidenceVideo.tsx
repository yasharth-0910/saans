"use client";

import dynamic from "next/dynamic";

// The SAM player decodes video with browser APIs, so it is only ever loaded in the browser.
const EvidenceVideo = dynamic(() => import("./SegmentedVideo"), {
  ssr: false,
  loading: () => <div className="mt-3 aspect-video w-full animate-pulse rounded-lg bg-zinc-100" />,
});

export default EvidenceVideo;
