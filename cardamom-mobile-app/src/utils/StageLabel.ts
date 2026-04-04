// Human-readable stage labels.
export const STAGE_LABELS: Record<number, string> = {
  0: 'Stage 0 – Healthy (no lesion)',
  1: 'Stage 1 – Mild (1–10 %)',
  2: 'Stage 2 – Moderate (11–25 %)',
  3: 'Stage 3 – Severe (26–50 %)',
  4: 'Stage 4 – Very Severe (> 50 %)',
};

// Background and text colors for each stage badge (React Native values).
export const STAGE_BADGE_COLORS: Record<number, { bg: string; text: string }> = {
  0: { bg: '#d1fae5', text: '#065f46' }, // emerald
  1: { bg: '#e0f2fe', text: '#0c4a6e' }, // sky
  2: { bg: '#fef3c7', text: '#78350f' }, // amber
  3: { bg: '#fee2e2', text: '#7f1d1d' }, // red
  4: { bg: '#ffe4e6', text: '#4c0519' }, // rose
};
