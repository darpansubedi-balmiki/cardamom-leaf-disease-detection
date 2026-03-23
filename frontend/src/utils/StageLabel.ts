// Human-readable stage labels.
export const STAGE_LABELS: Record<number, string> = {
    0: "Stage 0 – Healthy (no lesion)",
    1: "Stage 1 – Mild (1–10 %)",
    2: "Stage 2 – Moderate (11–25 %)",
    3: "Stage 3 – Severe (26–50 %)",
    4: "Stage 4 – Very Severe (> 50 %)",
};

// Tailwind classes for each stage badge.
export const STAGE_BADGE_TW: Record<number, string> = {
    0: "bg-emerald-100 text-emerald-800",
    1: "bg-sky-100 text-sky-800",
    2: "bg-amber-100 text-amber-900",
    3: "bg-red-100 text-red-900",
    4: "bg-rose-200 text-rose-950",
};