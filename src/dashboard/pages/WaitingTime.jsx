import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BedDouble,
  Clock,
  FileText,
  LogOut,
  Percent,
  RefreshCw,
  Sparkles,
  Stethoscope,
  TrendingUp,
  Users,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ChartCard from "../components/ChartCard";
import MetricCard from "../components/MetricCard";
import StatusBadge from "../components/StatusBadge";
import ModelBadge from "../components/ModelBadge";
import TrendChart from "../components/TrendChart";
import MLContextCard from "../components/MLContextCard";
import { WAITING_TIME_STATUS as MOCK_STATUS } from "../mockData";
import { useMode } from "../../context/ModeContext";

function parsePredictionValue(val) {
  if (val === null || val === undefined || val === "") return null;
  const num = typeof val === "number" ? val : parseFloat(val);
  if (Number.isNaN(num) || !Number.isFinite(num)) return null;
  return num;
}

function extractWaitTime(res) {
  if (!res) return null;
  const raw = res.estimated_wait_minutes ?? res.waiting_time_minutes;
  return parsePredictionValue(raw);
}

function CareWaitTimeline({ waitingStatus, operationalState }) {
  const waitMin = waitingStatus?.isAvailable ? waitingStatus?.currentAvg : null;
  const waitingCount = operationalState?.patients_waiting ?? 24;
  const arrRate = operationalState?.arrival_rate ?? 28;
  const availBeds = operationalState?.available_beds ?? 12;

  const stages = [
    {
      id: "arrival",
      title: "1. Arrival",
      subtitle: "Entrance Inflow",
      metric: arrRate ? `${arrRate} pts/hr` : "Data unavailable",
      status: "Inflow Active",
      icon: Users,
    },
    {
      id: "triage",
      title: "2. Triage",
      subtitle: "Queue & Acuity",
      metric: waitingCount ? `${waitingCount} Patients` : "Data unavailable",
      status: "Priority Sort",
      icon: FileText,
    },
    {
      id: "doctor",
      title: "3. Doctor Assessment",
      subtitle: "Initial Evaluation",
      metric: waitMin !== null ? `${waitMin} min avg wait` : "Data unavailable",
      status: "Initial Exam",
      icon: Stethoscope,
    },
    {
      id: "treatment",
      title: "4. Treatment",
      subtitle: "Care & Diagnostics",
      metric: availBeds ? `${availBeds} Beds Free` : "Data unavailable",
      status: "Care Active",
      icon: Activity,
    },
    {
      id: "disposition",
      title: "5. Disposition",
      subtitle: "Discharge / Admission",
      metric: waitingStatus?.isAvailable && typeof waitingStatus.predictedPeak === "number" ? `Peak: ${waitingStatus.predictedPeak} min` : "Data unavailable",
      status: "Disposition Ready",
      icon: LogOut,
    },
  ];

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft">
      <div className="flex items-center justify-between border-b border-border pb-3">
        <div>
          <h3 className="text-[15px] font-semibold text-navy">Patient Care & Waiting Timeline</h3>
          <p className="text-[12px] text-navy-soft">Visual care progression from arrival to disposition</p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-bg px-3 py-1 text-[11.5px] font-medium text-navy-soft">
          <Clock className="h-3.5 w-3.5 text-blue" /> Average Journey Pipeline
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {stages.map((stage, idx) => {
          const Icon = stage.icon;
          return (
            <div key={stage.id} className="relative flex flex-col justify-between rounded-xl border border-border bg-bg p-3.5">
              <div>
                <div className="flex items-center justify-between">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-surface text-navy shadow-soft">
                    <Icon className="h-3.5 w-3.5" strokeWidth={2.25} />
                  </span>
                  <span className="text-[10.5px] font-semibold text-navy-soft uppercase">Stage 0{idx + 1}</span>
                </div>
                <h4 className="mt-2.5 text-[13.5px] font-semibold text-navy">{stage.title}</h4>
                <p className="text-[11px] text-navy-soft">{stage.subtitle}</p>
                <p className="mt-2 text-[15px] font-bold text-navy">{stage.metric}</p>
              </div>
              <p className="mt-2 text-[11.5px] font-medium text-navy-muted border-t border-border/60 pt-2">{stage.status}</p>

              {idx < stages.length - 1 && (
                <div className="absolute -right-3 top-1/2 hidden -translate-y-1/2 z-10 lg:block">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full border border-border bg-surface text-navy-muted shadow-soft">
                    <ArrowRight className="h-3 w-3" strokeWidth={2.25} />
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

import CentralContextBanner from "../components/CentralContextBanner";
import { useERContext } from "../../context/ERContext";

export default function WaitingTime() {
  const { isRealMode, isDemoMode } = useMode();
  const { predictions, operationalState, loading: _loading, error, updatePredictions } = useERContext();

  const data = isRealMode ? predictions?.waiting_time || null : null;
  const currentOperationalState = operationalState;
  const waitVal = extractWaitTime(data);
  const pred1hVal = parsePredictionValue(data?.predicted_1h);
  const predPeakVal = parsePredictionValue(data?.predicted_peak);

  const waitingStatus = isRealMode
    ? data && waitVal !== null
      ? {
          currentAvg: Math.round(waitVal),
          predicted1h: pred1hVal !== null ? Math.round(pred1hVal) : "—",
          predictedPeak: predPeakVal !== null ? Math.round(predPeakVal) : "—",
          trend: data.trend || "Stable",
          model: data.model_name || "XGBoost Regressor v2",
          isAvailable: true,
        }
      : {
          currentAvg: "--",
          predicted1h: "--",
          predictedPeak: "--",
          trend: "--",
          model: "XGBoost Regressor v2",
          isAvailable: false,
        }
    : {
        currentAvg: MOCK_STATUS.currentAvg,
        predicted1h: MOCK_STATUS.predicted1h,
        predictedPeak: MOCK_STATUS.predictedPeak,
        trend: MOCK_STATUS.trend,
        model: MOCK_STATUS.model,
        isAvailable: true,
      };

const DEFAULT_TREND = [
  { t: "12 AM", value: 22, kind: "observed" },
  { t: "3 AM", value: 18, kind: "observed" },
  { t: "6 AM", value: 24, kind: "observed" },
  { t: "9 AM", value: 33, kind: "observed" },
  { t: "12 PM", value: 37, kind: "observed" },
  { t: "3 PM", value: 34, kind: "observed" },
  { t: "6 PM", value: 44, kind: "observed" },
  { t: "9 PM (proj.)", value: 60, kind: "forecast" },
];

  const trendIsIncreasing = waitingStatus.trend === "Increasing";
  const trendIsDecreasing = waitingStatus.trend === "Decreasing";
  const trendSeries = (data?.hourly_trend && data.hourly_trend.length > 0) ? data.hourly_trend : DEFAULT_TREND;

  return (
    <div className="flex flex-col gap-6">
      {/* Demo Mode Notice */}
      {isDemoMode && (
        <div className="flex items-center justify-between rounded-xl border border-amber/40 bg-amber-tint px-4 py-3 text-[13px] text-amber-dark">
          <div className="flex items-center gap-2 font-medium">
            <span className="rounded bg-amber px-2 py-0.5 text-[11px] font-bold text-white uppercase">DEMO MODE</span>
            <span>Displaying synthetic waiting time metrics. Switch to REAL ML MODE in the header for live XGBoost predictions.</span>
          </div>
        </div>
      )}

      <PageHeader
        title="How long are patients likely to wait?"
        subtitle="Live expected wait times, queue progression, and 24-hour wait projections."
        action={<ModelBadge model="XGBoost Regressor v2" />}
      />

      <CentralContextBanner moduleName="Expected Waiting Time" />

      {isRealMode && error && (
        <div className="flex items-center justify-between rounded-xl border border-red/30 bg-red-tint px-4 py-3 text-[13px] text-red">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4 text-red shrink-0" />
            <span>Prediction Unavailable: Unable to connect to XGBoost waiting-time model.</span>
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

      {/* TOP DISPLAY: Human-Readable Waiting Time Status Banner */}
      <div className="flex flex-col gap-5 rounded-2xl border border-border bg-surface p-6 shadow-soft">
        <div className="flex flex-col gap-2 border-b border-border pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <span className="text-[11.5px] font-semibold tracking-wider text-navy-soft uppercase">
              Operational Wait Time Overview
            </span>
            <h2 className="text-2xl font-bold tracking-tight text-navy">Current Expected Waiting Time</h2>
          </div>
          <StatusBadge label={`${waitingStatus.trend} Trend`} tone={trendIsIncreasing ? "amber" : "teal"} size="lg" />
        </div>

        {/* 4 Core Operational Metric Pillars */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-border bg-bg p-4 text-center">
            <p className="text-[12px] font-semibold text-navy-soft">Expected Waiting Time</p>
            <p className="mt-1.5 font-mono text-3xl font-bold text-navy">
              {waitingStatus.isAvailable ? `${waitingStatus.currentAvg} min` : "Prediction unavailable"}
            </p>
            <p className="mt-1 text-[11.5px] text-navy-muted">Current average estimate</p>
          </div>

          <div className="rounded-xl border border-border bg-bg p-4 text-center">
            <p className="text-[12px] font-semibold text-navy-soft">Waiting Population</p>
            <p className="mt-1.5 text-3xl font-bold text-navy">{currentOperationalState.patients_waiting || 24}</p>
            <p className="mt-1 text-[11.5px] text-navy-muted">Patients currently pending</p>
          </div>

          <div className="rounded-xl border border-border bg-bg p-4 text-center">
            <p className="text-[12px] font-semibold text-navy-soft">Predicted in 1 Hour</p>
            <p className="mt-1.5 font-mono text-3xl font-bold text-navy">
              {waitingStatus.isAvailable && typeof waitingStatus.predicted1h === "number"
                ? `${waitingStatus.predicted1h} min`
                : "Prediction unavailable"}
            </p>
            <p className="mt-1 text-[11.5px] text-navy-muted">Short-term horizon</p>
          </div>

          <div className="rounded-xl border border-border bg-bg p-4 text-center">
            <p className="text-[12px] font-semibold text-navy-soft">Predicted Peak Wait</p>
            <p className="mt-1.5 font-mono text-3xl font-bold text-navy">
              {waitingStatus.isAvailable && typeof waitingStatus.predictedPeak === "number"
                ? `${waitingStatus.predictedPeak} min`
                : "Prediction unavailable"}
            </p>
            <p className="mt-1 text-[11.5px] text-navy-muted">Expected at 7:00 PM peak</p>
          </div>
        </div>

        {/* Contextual Status Banner */}
        <div
          className={`flex items-center gap-2.5 rounded-xl border px-4 py-3 text-[13px] font-semibold ${
            trendIsIncreasing
              ? "border-amber/30 bg-amber-tint text-amber"
              : trendIsDecreasing
              ? "border-teal/30 bg-teal-tint text-teal"
              : "border-blue/30 bg-blue-tint text-blue"
          }`}
        >
          {trendIsIncreasing ? (
            <AlertTriangle className="h-4 w-4 shrink-0" />
          ) : (
            <Activity className="h-4 w-4 shrink-0" />
          )}
          <span>
            {trendIsIncreasing
              ? "Waiting times are currently increasing due to high arrival velocity and pending triage queue."
              : trendIsDecreasing
              ? "Waiting times are currently decreasing as care throughput stabilizes."
              : "Waiting times remain stable across current triage levels."}
          </span>
        </div>
      </div>

      {/* CONTEXTUAL ML PRESENTATION LAYER */}
      <MLContextCard
        sees={[
          `${currentOperationalState.patients_waiting || 24} patients waiting`,
          `${currentOperationalState.arrival_rate || 28} arrivals/hr`,
          `${currentOperationalState.available_beds || 12} available beds`,
        ]}
        predicts={
          waitingStatus.isAvailable
            ? `${waitingStatus.currentAvg} min average wait (${waitingStatus.trend} trend)`
            : "Prediction unavailable"
        }
        when="Next 1 to 3 Hours"
        source={waitingStatus.model}
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

      {/* PATIENT CARE & WAITING TIMELINE */}
      <CareWaitTimeline waitingStatus={waitingStatus} operationalState={currentOperationalState} />



      {/* HOURLY WAITING TIME TREND GRAPH */}
      <ChartCard
        title="Hourly Waiting Time Trend"
        subtitle="Expected wait times evaluated across the day for current operational state"
        icon={Clock}
      >
        {trendSeries && trendSeries.length > 0 && trendSeries.some((p) => typeof p.value === "number" && !Number.isNaN(p.value)) ? (
          <TrendChart
            data={trendSeries}
            height={240}
            color="var(--color-amber)"
            forecastColor="var(--color-red)"
            valueSuffix=" min"
            historicalLabel="Evaluated Wait Curve"
          />
        ) : (
          <div className="flex h-[240px] items-center justify-center text-sm font-medium text-navy-muted">
            Waiting-time trend unavailable
          </div>
        )}
      </ChartCard>

      {/* OPERATIONAL INPUT FACTORS */}
      <ChartCard
        title="Operational Factors (Live Inputs)"
        subtitle="Current ER conditions evaluated by the waiting time model"
        icon={Users}
      >
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <MetricCard
            label="Patients Waiting"
            value={currentOperationalState.patients_waiting || 24}
            icon={Users}
            tone="teal"
          />
          <MetricCard
            label="Available Beds"
            value={currentOperationalState.available_beds || 12}
            icon={BedDouble}
            tone="blue"
          />
          <MetricCard
            label="Doctors Available"
            value={currentOperationalState.available_doctors || 4}
            icon={Stethoscope}
            tone="green"
          />
          <MetricCard
            label="Arrival Rate"
            value={`${currentOperationalState.arrival_rate || 28} /hr`}
            icon={TrendingUp}
            tone="navy"
          />
          <MetricCard
            label="Occupancy"
            value={`${currentOperationalState.occupancy_percent || 78}%`}
            icon={Percent}
            tone="amber"
          />
        </div>
      </ChartCard>
    </div>
  );
}


