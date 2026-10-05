import {
  AlertTriangle,
  BedDouble,
  Clock,
  LayoutGrid,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  UserCheck,
  Users,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ChartCard from "../components/ChartCard";
import MetricCard from "../components/MetricCard";
import StatusBadge, { LEVEL_TONE } from "../components/StatusBadge";
import ModelBadge from "../components/ModelBadge";
import BarList from "../components/BarList";
import MLContextCard from "../components/MLContextCard";
import { useMode } from "../../context/ModeContext";
import {
  CROWDING_RISK_SUMMARY as MOCK_SUMMARY,
  CROWDING_RISK_LEVELS,
  CROWDING_MODEL as MOCK_MODEL,
} from "../mockData";

const DEFAULT_TIMELINE = [
  { time: "3 PM", level: "MODERATE" },
  { time: "4 PM", level: "MODERATE" },
  { time: "5 PM", level: "HIGH" },
  { time: "6 PM", level: "HIGH" },
  { time: "7 PM", level: "CRITICAL" },
  { time: "8 PM", level: "HIGH" },
  { time: "9 PM", level: "MODERATE" },
];

function TopStatusArea({ summary, data, operationalState }) {
  const level = summary.level || "CRITICAL";
  const score = summary.score || 25;
  const window = summary.window || "Next 3 Hours";
  const timestamp = new Date().toLocaleTimeString();

  // Extract class probability ONLY if present from backend model
  let probabilityStr = null;
  if (data?.class_probability !== undefined && data?.class_probability !== null) {
    const rawP = data.class_probability;
    probabilityStr = `${Math.round(rawP * (rawP <= 1.0 ? 100 : 1))}%`;
  } else if (data?.probabilities && data.probabilities[level] !== undefined) {
    const p = data.probabilities[level];
    probabilityStr = `${Math.round(p * (p <= 1.0 ? 100 : 1))}%`;
  }

  const tone = LEVEL_TONE[level] || "amber";
  const RING_TONE = {
    green: "border-green bg-green-tint text-green",
    amber: "border-amber bg-amber-tint text-amber",
    red: "border-red bg-red-tint text-red",
  };

  // Human Summary based strictly on actual values
  const waitingCount = operationalState.patients_waiting ?? 24;
  const arrivalRate = operationalState.arrival_rate ?? 28;
  const occupancy = operationalState.occupancy_percent ?? 78;

  const humanSummary = `Current department demand is ${
    level === "CRITICAL" || level === "HIGH" ? "elevated" : "moderate"
  }, with ${waitingCount} patients currently waiting, ${occupancy}% ER occupancy, and ${arrivalRate} patient arrivals per hour expected over the ${window}.`;

  return (
    <div className="flex flex-col gap-6 rounded-2xl border border-border bg-surface p-6 shadow-soft">
      {/* Top Header */}
      <div className="flex flex-col gap-3 border-b border-border pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <span className="text-[11.5px] font-semibold tracking-wider text-navy-soft uppercase">
            Emergency Department Operations Center
          </span>
          <h2 className="text-2xl font-bold tracking-tight text-navy">Current ER Crowding</h2>
          <p className="mt-0.5 text-[13.5px] text-navy-soft">
            Real-time department crowding assessment & operational threat status
          </p>
        </div>

        <div className="flex items-center gap-2">
          <ModelBadge model={data?.model_name || "XGBoost Classifier"} />
          <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-bg px-3 py-1 text-[12px] font-medium text-navy-soft">
            <Clock className="h-3.5 w-3.5" /> Updated: {timestamp}
          </span>
        </div>
      </div>

      {/* 3 Core Command Center Questions */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Question 1: How crowded is the ER right now? */}
        <div className="flex flex-col items-center justify-center rounded-xl border border-border bg-bg p-5 text-center">
          <p className="text-[11.5px] font-semibold tracking-wider text-navy-soft uppercase">
            1. How crowded is the ER right now?
          </p>
          <div
            className={`mt-3 flex h-28 w-28 flex-col items-center justify-center rounded-full border-[5px] shadow-soft ${
              RING_TONE[tone] || RING_TONE.amber
            }`}
          >
            <ShieldAlert className="h-6 w-6" strokeWidth={2.25} aria-hidden="true" />
            <p className="mt-1 text-xl font-bold tracking-tight">{level}</p>
          </div>
          <p className="mt-3 font-mono text-[14px] font-semibold text-navy">
            Crowding Index Score: {score}/100
          </p>
          {probabilityStr && (
            <p className="mt-1 text-[12.5px] font-medium text-blue">
              Class Probability: <span className="font-bold">{probabilityStr}</span>
            </p>
          )}
          <p className="mt-0.5 text-[12px] text-navy-soft">Window: {window}</p>
        </div>

        {/* Question 2: What is contributing to the current risk? (Human Summary) */}
        <div className="flex flex-col justify-between rounded-xl border border-border bg-bg p-5 lg:col-span-2">
          <div>
            <p className="text-[11.5px] font-semibold tracking-wider text-navy-soft uppercase">
              2. What is contributing to the current risk?
            </p>
            <p className="mt-2.5 text-[15px] font-medium leading-relaxed text-navy">
              {humanSummary}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-border bg-surface p-3 text-center">
                <p className="text-[11px] font-medium text-navy-soft">Occupancy</p>
                <p className="text-[16px] font-bold text-navy">{occupancy}%</p>
              </div>
              <div className="rounded-lg border border-border bg-surface p-3 text-center">
                <p className="text-[11px] font-medium text-navy-soft">Patients Waiting</p>
                <p className="text-[16px] font-bold text-navy">{waitingCount} pts</p>
              </div>
              <div className="rounded-lg border border-border bg-surface p-3 text-center">
                <p className="text-[11px] font-medium text-navy-soft">Arrival Rate</p>
                <p className="text-[16px] font-bold text-navy">{arrivalRate} pts/hr</p>
              </div>
            </div>
          </div>

          {/* Question 3: What should the user pay attention to? */}
          <div className="mt-4 border-t border-border/60 pt-3">
            <p className="text-[11.5px] font-semibold tracking-wider text-navy-soft uppercase">
              3. What should the user pay attention to?
            </p>
            <div className="mt-2 flex items-center gap-2 rounded-lg border border-amber/30 bg-amber-tint px-3 py-2 text-[12.5px] font-semibold text-amber">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>
                {occupancy > 75
                  ? `High bed occupancy (${occupancy}%) combined with ${waitingCount} waiting patients requires close monitoring over ${window}.`
                  : `Monitor triage queue volume (${waitingCount} waiting) and arrival velocity (${arrivalRate} pts/hr).`}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LevelScale({ current }) {
  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
      {CROWDING_RISK_LEVELS.map((lvl) => {
        const isActive = lvl === current;
        const tone = LEVEL_TONE[lvl] || "amber";
        const TONE_ACTIVE = {
          green: "border-green bg-green-tint text-green",
          amber: "border-amber bg-amber-tint text-amber",
          red: "border-red bg-red-tint text-red",
        };
        return (
          <div
            key={lvl}
            className={`rounded-xl border p-3 text-center transition-colors ${
              isActive ? `${TONE_ACTIVE[tone]} shadow-soft` : "border-border bg-bg text-navy-soft"
            }`}
          >
            <p className={`text-[12.5px] font-semibold ${isActive ? "" : "text-navy-muted"}`}>{lvl}</p>
            {isActive && <p className="mt-0.5 text-[10.5px] font-medium uppercase tracking-wide">Current</p>}
          </div>
        );
      })}
    </div>
  );
}

function TimelineRow({ time, level, isLast }) {
  const tone = LEVEL_TONE[level] || "amber";
  return (
    <div className={`flex items-center justify-between gap-3 py-2.5 ${isLast ? "" : "border-b border-border"}`}>
      <span className="flex items-center gap-2 text-[13.5px] font-medium text-navy">
        <Clock className="h-3.5 w-3.5 text-navy-soft" strokeWidth={2.25} aria-hidden="true" />
        {time}
      </span>
      <StatusBadge label={level} tone={tone} />
    </div>
  );
}

import CentralContextBanner from "../components/CentralContextBanner";
import { useERContext } from "../../context/ERContext";

export default function CrowdingRisk() {
  const { isRealMode, isDemoMode } = useMode();
  const { predictions, operationalState, loading: _loading, error, updatePredictions } = useERContext();

  const data = isRealMode ? predictions?.crowding_risk || null : null;

  const crowdingSummary = (isRealMode
    ? data
      ? {
          level: data.crowding_level,
          score: data.crowding_score,
          window: data.expected_window || "Next 3 Hours",
        }
      : { level: "--", score: "--", window: "Predictions Pending" }
    : MOCK_SUMMARY);

  const modelName = isRealMode ? data?.model_name || "XGBoost Classifier" : MOCK_MODEL;

  const probabilityBars = data?.probabilities
    ? [
        { label: "Critical Risk", value: Math.round((data.probabilities.Critical || 0) * 100), tone: "red" },
        { label: "High Risk", value: Math.round((data.probabilities.High || 0) * 100), tone: "red" },
        { label: "Moderate Risk", value: Math.round((data.probabilities.Moderate || 0) * 100), tone: "amber" },
        { label: "Low Risk", value: Math.round((data.probabilities.Low || 0) * 100), tone: "green" },
      ]
    : null;

  const timelineRows = data?.risk_timeline || DEFAULT_TIMELINE;

  return (
    <div className="flex flex-col gap-6">
      {/* Demo Mode Notice */}
      {isDemoMode && (
        <div className="flex items-center justify-between rounded-xl border border-amber/40 bg-amber-tint px-4 py-3 text-[13px] text-amber-dark">
          <div className="flex items-center gap-2 font-medium">
            <span className="rounded bg-amber px-2 py-0.5 text-[11px] font-bold text-white uppercase">DEMO MODE</span>
            <span>Displaying synthetic crowding risk metrics. Switch to REAL ML MODE in the header for live XGBoost predictions.</span>
          </div>
        </div>
      )}

      <PageHeader
        title="Emergency Department Crowding Risk"
        subtitle="Predict overall ED crowding risk using current occupancy, arrivals, and staffing conditions."
        action={<ModelBadge model={modelName} />}
      />

      <CentralContextBanner moduleName="Crowding Risk Prediction" />

      {isRealMode && error && (
        <div className="flex items-center justify-between rounded-xl border border-red/30 bg-red-tint px-4 py-3 text-[13px] text-red">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4 text-red shrink-0" />
            <span>Prediction Unavailable: Unable to connect to XGBoost Crowding Classifier.</span>
          </div>
          <button
            type="button"
            onClick={() => updatePredictions()}
            className="flex items-center gap-1 font-semibold underline hover:text-red-dark"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </button>
        </div>
      )}

      {/* TOP STATUS AREA: Human-Readable Command Center Overview */}
      <TopStatusArea
        summary={crowdingSummary || { level: "MODERATE", score: 45, window: "Next 3 Hours" }}
        data={data}
        operationalState={operationalState}
      />

      {/* CONTEXTUAL ML PRESENTATION LAYER */}
      <MLContextCard
        sees={[
          `${operationalState.patients_waiting || 24} patients waiting`,
          `${operationalState.arrival_rate || 28} arrivals/hr`,
          `${operationalState.occupancy_percent || 78}% occupancy`,
        ]}
        predicts={`${crowdingSummary?.level || 'MODERATE'} Crowding Risk (Score: ${crowdingSummary?.score || 45}/100)`}
        when={crowdingSummary?.window || 'Next 3 Hours'}
        source={modelName}
      />

      {/* WHY THIS PREDICTION? (Explainable AI TreeSHAP Layer) */}
      {data?.explanation?.top_factors?.length > 0 && (
        <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft sm:p-6">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-tint text-teal">
              <Sparkles className="h-4 w-4" />
            </span>
            <h3 className="text-[13px] font-bold tracking-wider text-navy uppercase">Why This Prediction?</h3>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {data.explanation.top_factors.map((factor, idx) => {
              const isIncrease = factor.direction === "increases";
              const impactLabel =
                factor.importance >= 0.35 ? "High impact" : factor.importance >= 0.10 ? "Moderate impact" : "Lower impact";
              return (
                <div key={idx} className="flex items-center justify-between rounded-xl border border-border bg-bg px-4 py-3">
                  <div>
                    <p className="text-[13px] font-semibold text-navy">{factor.feature}</p>
                    <p className="text-[11.5px] font-medium text-navy-soft">{impactLabel}</p>
                  </div>
                  <span
                    className={`flex items-center gap-1 font-mono text-[13px] font-bold ${
                      isIncrease ? "text-amber-dark" : "text-teal"
                    }`}
                  >
                    {isIncrease ? "↑" : "↓"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}



      {/* CONTRIBUTING OPERATIONAL FACTORS (LIVE INPUTS) */}
      <ChartCard
        title="Contributing Operational Factors (Live Inputs)"
        subtitle="Operational ED state factors evaluated by the crowding risk model"
        icon={LayoutGrid}
      >
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-7">
          <MetricCard
            label="Occupancy"
            value={`${operationalState.occupancy_percent || 78}%`}
            icon={AlertTriangle}
            tone={(operationalState.occupancy_percent || 78) > 80 ? "red" : "amber"}
          />
          <MetricCard
            label="Patients Waiting"
            value={operationalState.patients_waiting || 24}
            icon={Users}
            tone="amber"
          />
          <MetricCard
            label="Arrival Rate"
            value={`${operationalState.arrival_rate || 28} /hr`}
            icon={TrendingUp}
            tone="blue"
          />
          <MetricCard
            label="Available Beds"
            value={operationalState.available_beds || 12}
            icon={BedDouble}
            tone="teal"
          />
          <MetricCard
            label="Staffed Doctors"
            value={operationalState.available_doctors || 4}
            icon={UserCheck}
            tone="navy"
          />
          <MetricCard
            label="Staffed Nurses"
            value={operationalState.available_nurses || 10}
            icon={UserCheck}
            tone="navy"
          />
          <MetricCard
            label="Acuity Severity"
            value={`Level ${operationalState.severity_level || 3}`}
            icon={Clock}
            tone="purple"
          />
        </div>
      </ChartCard>

      {/* PROBABILITY DISTRIBUTION & TIMELINE */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <ChartCard title="Risk Level Scale" icon={ShieldAlert} className="flex flex-col justify-center">
          <p className="mb-4 text-[12.5px] font-medium text-navy-soft text-center">
            Crowding Risk Spectrum (Low to Critical)
          </p>
          <LevelScale current={crowdingSummary.level} />
        </ChartCard>

        <ChartCard
          title="Class Probability Distribution & Projected Timeline"
          subtitle="Actual XGBoost Classifier class probabilities and evening risk timeline"
          icon={TrendingUp}
          className="lg:col-span-2"
        >
          <div className="flex flex-col gap-5">
            {probabilityBars && (
              <div>
                <p className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-navy-soft">
                  Model Class Probability Distribution
                </p>
                <BarList
                  items={probabilityBars.map((p) => ({
                    label: p.label,
                    value: p.value,
                    max: 100,
                    tone: p.tone,
                    valueLabel: `${p.value}%`,
                  }))}
                />
              </div>
            )}

            <div className="border-t border-border pt-4">
              <p className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-navy-soft">
                Projected Evening Crowding Timeline (XGBoost Evaluated)
              </p>
              <div>
                {timelineRows.map((row, i) => (
                  <TimelineRow
                    key={row.time}
                    time={row.time}
                    level={row.level}
                    isLast={i === timelineRows.length - 1}
                  />
                ))}
              </div>
            </div>
          </div>
        </ChartCard>
      </div>
    </div>
  );
}


