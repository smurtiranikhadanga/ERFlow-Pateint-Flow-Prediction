import { ArrowUpRight, ArrowDownRight, HelpCircle } from "lucide-react";

const TONE = {
  blue: { icon: "text-blue", bg: "bg-blue-tint" },
  teal: { icon: "text-teal", bg: "bg-teal-tint" },
  green: { icon: "text-green", bg: "bg-green-tint" },
  amber: { icon: "text-amber", bg: "bg-amber-tint" },
  red: { icon: "text-red", bg: "bg-red-tint" },
};

export default function SummaryCard({ label, value, trend, trendDirection, tone = "blue", icon: Icon, onExplain }) {
  const t = TONE[tone] || TONE.blue;
  const TrendIcon = trendDirection === "down" ? ArrowDownRight : ArrowUpRight;

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft">
      <div className="flex items-start justify-between gap-2">
        <p className="text-[13px] font-medium uppercase tracking-wide text-navy-soft min-w-0 flex-1">{label}</p>
        <div className="flex items-center gap-1.5 shrink-0">
          {onExplain && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onExplain();
              }}
              className="inline-flex items-center gap-1 rounded-md border border-blue/30 bg-blue-tint px-2.5 py-1 text-[11.5px] font-bold text-blue hover:bg-blue hover:text-white transition-all cursor-pointer z-10 shadow-sm"
              title="Click to view TreeSHAP feature attributions"
            >
              <HelpCircle className="h-3.5 w-3.5 shrink-0" /> Why?
            </button>
          )}
          {Icon && (
            <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${t.bg} ${t.icon}`}>
              <Icon className="h-4 w-4" strokeWidth={2.25} aria-hidden="true" />
            </span>
          )}
        </div>
      </div>
      <p className="mt-3 font-mono text-3xl font-semibold text-navy sm:text-[2rem]">{value}</p>
      {trend && (
        <p
          className={`mt-2 inline-flex items-center gap-1 text-[12.5px] font-medium ${
            trendDirection === "down" ? "text-green" : "text-navy-soft"
          }`}
        >
          <TrendIcon className="h-3.5 w-3.5" strokeWidth={2.5} aria-hidden="true" />
          {trend}
        </p>
      )}
    </div>
  );
}
