// Inlined at build time. Set it in .env.local for dev and in the Vercel project for deploys.
if (!process.env.NEXT_PUBLIC_API_URL) throw new Error("NEXT_PUBLIC_API_URL is not set");
export const API_URL = process.env.NEXT_PUBLIC_API_URL.replace(/\/+$/, "");

export type Rule = {
  id: string;
  stage: number; // GRAP stage the rule applies from, 1 to 4
  title: string;
  rule_text: string;
  violation_types: string[];
  source: string;
};

export type Finding = {
  violation_type: string;
  rule_id: string;
  rule: Rule | null;
  confidence: number;
  frame_index: number;
  box: [number, number, number, number];
  description: string;
};

export type Report = {
  id: string;
  created_at: string;
  lat: number;
  lng: number;
  media_type: "image" | "video" | "seed";
  summary: string;
  findings: Finding[];
  frames: string[];
};

export type Hotspot = { lat: number; lng: number; count: number; report_ids: string[] };

export const stageName = (stage: number) => ["I", "II", "III", "IV"][stage - 1] ?? String(stage);

export const label = (violationType: string) =>
  violationType.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
