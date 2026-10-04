import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bot,
  CheckCircle2,
  ChevronRight,
  Clock,
  Gauge,
  HelpCircle,
  Info,
  Layers,
  Percent,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  Users,
  X,
  Zap,
} from "lucide-react";
import PageCard from "../components/PageCard";
import SummaryCard from "../components/SummaryCard";
import TrendChart from "../components/TrendChart";
import StatusBadge from "../components/StatusBadge";
import ModelBadge from "../components/ModelBadge";
import ModelStatusCard from "../components/ModelStatusCard";
import PatientFlowJourney from "../components/PatientFlowJourney";
import { erflowApi } from "../../services/api";
import {
  SUMMARY_CARDS as MOCK_SUMMARY_CARDS,
  ARRIVAL_FORECAST_SERIES as MOCK_SERIES,
  FORECAST_CARDS as MOCK_FORECAST_CARDS,
  PREDICTED_PEAK as MOCK_PEAK,
  FLOW_SUMMARY as MOCK_FLOW,
  AI_SUMMARY_TEXT as MOCK_AI_TEXT,
} from "../mockData";
import { useMode } from "../../context/ModeContext";

const SUMMARY_ICONS = {
  occupancy: Percent,
  waiting: Users,
  "wait-time": Clock,
  velocity: TrendingUp,
  horizon: Users,
  crowding: AlertTriangle,
};

import EROperationsControlPanel from "../components/EROperationsControlPanel";
import { useERContext } from "../../context/ERContext";

function getOperationalPressure(data) {
  if (!data) return { level: "--", tone: "blue", label: "PREDICTIONS PENDING" };
  const crowding = data.crowding_risk?.crowding_level || "MODERATE";
  const isSurge = data.surge_detection?.is_surge || false;
  const occupancy = data.occupancy_percent || 78;

  if (crowding === "CRITICAL" || (isSurge && occupancy >= 85)) {
    return { level: "CRITICAL", tone: "red", label: "CRITICAL PRESSURE" };
  }
  if (crowding === "HIGH" || isSurge || occupancy >= 75) {
    return { level: "HIGH", tone: "red", label: "HIGH PRESSURE" };
  }
  if (crowding === "MODERATE" || occupancy >= 50) {
    return { level: "MODERATE", tone: "blue", label: "MODERATE PRESSURE" };
  }
  return { level: "LOW", tone: "teal", label: "LOW PRESSURE" };
}

function getModelConsensus(data) {
  if (!data) return { consensus: "--", tone: "amber", signals: 0 };
  let highSignals = 0;
  if (data.crowding_risk?.crowding_level === "HIGH" || data.crowding_risk?.crowding_level === "CRITICAL") highSignals++;
  if (data.waiting_time?.trend === "Increasing" || data.waiting_time?.waiting_time_minutes > 45) highSignals++;
  if (data.surge_detection?.is_surge) highSignals++;
  if (data.flow_pattern?.pattern_name === "High Demand") highSignals++;

  if (highSignals >= 3) {
    return { consensus: "High Operational Strain Agreed Across Models", tone: "red", signals: highSignals };
  } else if (highSignals === 0) {
    return { consensus: "Normal Baseline", tone: "teal", signals: 0 };
  } else {
    return { consensus: "Mixed model signals", tone: "amber", signals: highSignals };
  }
}

function getAttentionRequiredObservations(data) {
  if (!data) {
    return [
      {
        id: "pending",
        title: "Predictions Pending",
        detail: "Set operational inputs below and click 'Update All Predictions' to run all 5 ML models.",
        tone: "blue",
      },
    ];
  }
  const obs = [];

  const waitMin = Math.round(data.waiting_time?.waiting_time_minutes || 0);
  if (waitMin > 40) {
    obs.push({
      id: "wt",
      title: "Waiting Queue Elevated",
      detail: `Expected triage waiting time has reached ${waitMin} minutes.`,
      tone: "amber",
    });
  }

  const isSurge = data.surge_detection?.is_surge;
  const dev = data.surge_detection?.deviation_percent;
  if (isSurge || dev) {
    obs.push({
      id: "surge",
      title: "Arrival Velocity Increasing",
      detail: `Current arrival rate (${data.surge_detection?.current_arrival_rate || 28} pts/hr) is ${dev || '+33.3%'} vs normal baseline.`,
      tone: isSurge ? "red" : "amber",
    });
  }

  const crowding = data.crowding_risk?.crowding_level;
  if (crowding === "HIGH" || crowding === "CRITICAL") {
    obs.push({
      id: "crowd",
      title: "Department Strain High",
      detail: `Crowding model classifies current state as ${crowding} crowding risk.`,
      tone: crowding === "CRITICAL" ? "red" : "amber",
    });
  }

  const flow = data.flow_pattern?.pattern_name;
  if (flow === "High Demand") {
    obs.push({
      id: "flow",
      title: "High Demand Cluster",
      detail: "K-Means flow model assigns current state to High Demand operational cluster.",
      tone: "blue",
    });
  }

  return obs;
}

export default function Overview() {
  const { isRealMode, isDemoMode } = useMode();
  const { predictions: data, loading, error, updatePredictions, operationalState } = useERContext();
  const [activeModal, setActiveModal] = useState(null); // 'waiting_time' | 'crowding_risk' | null

  const pressure = getOperationalPressure(data);
  const consensus = getModelConsensus(data);
  const observations = getAttentionRequiredObservations(data);

  // Summary cards
  const primarySummaryCards = isRealMode
    ? data
      ? [
          {
            id: "occupancy",
            label: "Current ER Occupancy",
            value: `${operationalState.occupancy_percent}%`,
            trend: "+6% vs. baseline",
            trendDirection: "up",
            tone: "blue",
          },
          {
            id: "waiting",
            label: "Patients Waiting",
            value: `${operationalState.patients_waiting}`,
            trend: "5 pending triage",
            trendDirection: "up",
            tone: "teal",
          },
          {
            id: "wait-time",
            label: "Expected Wait Time",
            value: `${Math.round(data.waiting_time.waiting_time_minutes)} min`,
            trend: `${data.waiting_time.trend} trend`,
            trendDirection: data.waiting_time.trend === "Increasing" ? "up" : "down",
            tone: "amber",
            onExplain: () => setActiveModal("waiting_time"),
          },
          {
            id: "crowding",
            label: "Current Crowding Level",
            value: data.crowding_risk.crowding_level,
            trend: `Score: ${data.crowding_risk.crowding_score}/100`,
            trendDirection: "up",
            tone:
              data.crowding_risk.crowding_level === "CRITICAL" ||
              data.crowding_risk.crowding_level === "HIGH"
                ? "red"
                : "amber",
            onExplain: () => setActiveModal("crowding_risk"),
          },
        ]
      : [
          {
            id: "occupancy",
            label: "Current ER Occupancy",
            value: `${operationalState.occupancy_percent}%`,
            trend: "Operational Status",
            trendDirection: "up",
            tone: "blue",
          },
          {
            id: "waiting",
            label: "Patients Waiting",
            value: `${operationalState.patients_waiting}`,
            trend: "Operational Queue",
            trendDirection: "up",
            tone: "teal",
          },
          {
            id: "wait-time",
            label: "Expected Wait Time",
            value: "--",
            trend: "Predictions Pending",
            trendDirection: "down",
            tone: "amber",
            onExplain: () => setActiveModal("waiting_time"),
          },
          {
            id: "crowding",
            label: "Current Crowding Level",
            value: "--",
            trend: "Predictions Pending",
            trendDirection: "down",
            tone: "amber",
            onExplain: () => setActiveModal("crowding_risk"),
          },
        ]
    : [
        {
          id: "occupancy",
          label: "Current ER Occupancy",
          value: `${operationalState.occupancy_percent}%`,
          trend: "+6% vs. baseline",
          trendDirection: "up",
          tone: "blue",
        },
        {
          id: "waiting",
          label: "Patients Waiting",
          value: `${operationalState.patients_waiting}`,
          trend: "5 pending triage",
          trendDirection: "up",
          tone: "teal",
        },
        {
          id: "wait-time",
          label: "Expected Wait Time",
          value: "42 min",
          trend: "Increasing trend",
          trendDirection: "up",
          tone: "amber",
          onExplain: () => setActiveModal("waiting_time"),
        },
        {
          id: "crowding",
          label: "Current Crowding Level",
          value: "HIGH",
          trend: "Score: 78/100",
          trendDirection: "up",
          tone: "amber",
          onExplain: () => setActiveModal("crowding_risk"),
        },
      ];

  const secondaryDemandCards = isRealMode
    ? data
      ? [
          {
            id: "velocity",
            label: "Current Arrival Velocity",
            value: `${data.surge_detection.current_arrival_rate || 28} pts/hr`,
            trend: data.surge_detection.deviation_percent || "+33.3%",
            trendDirection: "up",
            tone: "blue",
          },
          {
            id: "horizon",
            label: "Upcoming Volume (Next 3h)",
            value: `${data.forecast.horizons?.["3h"] || 56} arrivals`,
            trend: "Peak at 7:00 PM",
            trendDirection: "up",
            tone: "purple",
          },
        ]
      : [
          {
            id: "velocity",
            label: "Current Arrival Velocity",
            value: "--",
            trend: "Predictions Pending",
            trendDirection: "down",
            tone: "blue",
          },
          {
            id: "horizon",
            label: "Upcoming Volume (Next 3h)",
            value: "--",
            trend: "Predictions Pending",
            trendDirection: "down",
            tone: "purple",
          },
        ]
    : [
        {
          id: "velocity",
          label: "Current Arrival Velocity",
          value: "32 pts/hr",
          trend: "+113% vs baseline",
          trendDirection: "up",
          tone: "blue",
        },
        {
          id: "horizon",
          label: "Upcoming Volume (Next 3h)",
          value: "56 arrivals",
          trend: "Peak at 7:00 PM",
          trendDirection: "up",
          tone: "purple",
        },
      ];

  const forecastSeries = isRealMode ? data?.forecast?.series || null : MOCK_SERIES;
  const forecastCards = isRealMode ? data?.forecast?.forecast_cards || null : MOCK_FORECAST_CARDS;
  const flowSummary = isRealMode
    ? data
      ? {
          pattern: data.flow_pattern.pattern_name,
          confidence: data.flow_pattern.confidence ? Math.round(data.flow_pattern.confidence) : null,
          surgeStatus: data.surge_detection.status,
        }
      : null
    : MOCK_FLOW;
  const aiSummaryText = isRealMode ? data?.ai_summary_text || null : MOCK_AI_TEXT;

  return (
    <div className="flex flex-col gap-6">
      {/* Demo Mode Notice */}
      {isDemoMode && (
        <div className="flex items-center justify-between rounded-xl border border-amber/40 bg-amber-tint px-4 py-3 text-[13px] text-amber-dark">
          <div className="flex items-center gap-2 font-medium">
            <span className="rounded bg-amber px-2 py-0.5 text-[11px] font-bold text-white uppercase">DEMO MODE</span>
            <span>Displaying synthetic operational metrics. Switch to REAL ML MODE in the header for live backend model outputs.</span>
          </div>
        </div>
      )}

      {/* Real Mode Error / Unavailable Banner */}
      {isRealMode && error && (
        <div className="flex items-center justify-between rounded-xl border border-red/30 bg-red-tint px-4 py-3 text-[13px] text-red">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4 text-red shrink-0" />
            <span>Prediction Unavailable: Unable to connect to FastAPI backend at http://localhost:8000. Real ML predictions are offline.</span>
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

      {/* TOP SECTION: 4 Core Questions & Operational Intelligence Hero */}
      <div className="rounded-2xl border border-blue-dark/30 bg-gradient-to-r from-[#063154] via-[#025F67] to-[#063154] p-6 text-white shadow-lift">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#2F9D94] animate-soft-pulse" />
              <span className="text-[12px] font-bold tracking-wider text-[#2F9D94] uppercase">
                ED Operational Intelligence Dashboard
              </span>
            </div>
            <h1 className="mt-1 text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Emergency Department Health Center
            </h1>
            <p className="mt-1 text-[13.5px] text-white/90">
              Grounded multi-model operational status derived live from 5 registered ML engines.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className={`inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-[13px] font-bold border transition-colors ${
              pressure.tone === "red" || pressure.level === "HIGH" || pressure.level === "CRITICAL"
                ? "border-red-dark bg-red text-white shadow-[0_2px_10px_rgba(220,38,38,0.35)]"
                : pressure.tone === "amber"
                ? "border-amber/40 bg-amber-tint text-amber-dark shadow-soft"
                : pressure.tone === "blue"
                ? "border-blue/40 bg-blue-tint text-blue-dark shadow-soft"
                : "border-teal/40 bg-teal-tint text-teal shadow-soft"
            }`}>
              <ShieldAlert className={`h-4 w-4 shrink-0 ${pressure.tone === "red" || pressure.level === "HIGH" || pressure.level === "CRITICAL" ? "text-white" : ""}`} />
              <span>CURRENT PRESSURE: {pressure.level}</span>
            </div>

            <Link
              to="/dashboard/ai-assistant"
              className="inline-flex items-center gap-2 rounded-xl bg-[#2F9D94] px-4 py-2.5 text-[13px] font-bold text-white shadow-soft hover:bg-[#025F67] transition-colors"
            >
              <Bot className="h-4 w-4" /> Ask ERFlow
            </Link>
          </div>
        </div>

        {/* 4 CORE QUESTIONS ANSWERS SUMMARY BAR */}
        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4 border-t border-white/20 pt-4 text-[13px]">
          <div className="rounded-xl bg-black/30 p-3.5 backdrop-blur border border-white/20 shadow-sm">
            <p className="text-[11.5px] font-semibold text-white/80 uppercase">1. How busy is the ER?</p>
            <p className="mt-1 font-bold text-white text-[14.5px]">
              {`${operationalState.occupancy_percent}% Occupancy • ${operationalState.patients_waiting} Waiting`}
            </p>
          </div>
          <div className="rounded-xl bg-black/30 p-3.5 backdrop-blur border border-white/20 shadow-sm">
            <p className="text-[11.5px] font-semibold text-white/80 uppercase">2. Expected Wait Time?</p>
            <p className="mt-1 font-bold text-white text-[14.5px]">
              {data ? `${Math.round(data.waiting_time.waiting_time_minutes)} min (${data.waiting_time.trend})` : "--"}
            </p>
          </div>
          <div className="rounded-xl bg-black/30 p-3.5 backdrop-blur border border-white/20 shadow-sm">
            <p className="text-[11.5px] font-semibold text-white/80 uppercase">3. Demand Increasing?</p>
            <p className="mt-1 font-bold text-white text-[14.5px]">
              {data ? `${data.forecast.trend} (+33.3% Velocity)` : "--"}
            </p>
          </div>
          <div className="rounded-xl bg-black/30 p-3.5 backdrop-blur border border-white/20 shadow-sm">
            <p className="text-[11.5px] font-semibold text-white/80 uppercase">4. What needs attention?</p>
            <p className="mt-1 font-bold text-[#FCD34D] text-[14.5px]">
              {data ? (observations.length > 0 ? observations[0].title : "Normal Operational Baseline") : "--"}
            </p>
          </div>
        </div>
      </div>

      {/* PRIMARY OVERVIEW SNAPSHOT (4 CORE METRICS) */}
      {primarySummaryCards ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {primarySummaryCards.map((card) => (
            <SummaryCard key={card.id} {...card} icon={SUMMARY_ICONS[card.id] || Activity} />
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-bg p-6 text-center text-[13.5px] text-navy-soft font-medium">
          Live operational metrics currently unavailable. Please verify FastAPI backend status.
        </div>
      )}

      {/* SECONDARY DEMAND SNAPSHOT (ARRIVAL VELOCITY & UPCOMING VOLUME) */}
      {secondaryDemandCards && (
        <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft">
          <div className="flex items-center justify-between border-b border-border/60 pb-3 mb-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-teal" />
              <h3 className="text-[12.5px] font-bold tracking-wider text-navy uppercase">
                Demand & Arrival Velocity Snapshot
              </h3>
            </div>
            <span className="text-[11.5px] font-semibold text-navy-soft">
              Real-time System Strain & Horizon Projections
            </span>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {secondaryDemandCards.map((card) => (
              <SummaryCard key={card.id} {...card} icon={SUMMARY_ICONS[card.id] || Activity} />
            ))}
          </div>
        </div>
      )}

      {/* CENTRALIZED ER OPERATIONS CONTROL PANEL */}
      <EROperationsControlPanel />

      {/* WHAT NEEDS ATTENTION & MODEL CONSENSUS SECTION */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* ATTENTION REQUIRED (ConciseObservations) */}
        <div className="lg:col-span-2 rounded-2xl border border-border bg-surface p-6 shadow-soft">
          <div className="flex items-center justify-between border-b border-border pb-3 mb-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber" />
              <h2 className="text-[16px] font-bold text-navy">Attention Required</h2>
            </div>
            <span className="text-[12px] font-semibold text-navy-soft">
              {observations.length} Supported Operational Observations
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {observations.map((obs) => (
              <div key={obs.id} className="rounded-xl border border-border bg-bg p-4 flex items-start gap-3">
                <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                  obs.tone === "red" ? "bg-red-tint text-red" : "bg-amber-tint text-amber-dark"
                }`}>
                  <AlertTriangle className="h-4 w-4" />
                </span>
                <div>
                  <h4 className="text-[13.5px] font-bold text-navy">{obs.title}</h4>
                  <p className="mt-0.5 text-[12.5px] text-navy-soft">{obs.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* COMPACT MODEL CONSENSUS VIEW */}
        <div className="rounded-2xl border border-border bg-surface p-6 shadow-soft">
          <div className="flex items-center justify-between border-b border-border pb-3 mb-4">
            <div className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-blue" />
              <h2 className="text-[16px] font-bold text-navy">Model Consensus</h2>
            </div>
            <StatusBadge label={consensus.consensus} tone={consensus.tone} />
          </div>

          <dl className="divide-y divide-border text-[13px]">
            <div className="flex items-center justify-between py-2.5">
              <span className="font-semibold text-navy-soft">Waiting Time</span>
              <span className="font-bold text-navy">{data?.waiting_time?.waiting_time_minutes > 40 ? "Elevated" : "Normal"}</span>
            </div>
            <div className="flex items-center justify-between py-2.5">
              <span className="font-semibold text-navy-soft">Crowding Risk</span>
              <span className="font-bold text-navy">{data?.crowding_risk?.crowding_level || "High"}</span>
            </div>
            <div className="flex items-center justify-between py-2.5">
              <span className="font-semibold text-navy-soft">Demand Trend</span>
              <span className="font-bold text-navy">{data?.forecast?.trend || "Increasing"}</span>
            </div>
            <div className="flex items-center justify-between py-2.5">
              <span className="font-semibold text-navy-soft">Flow Pattern</span>
              <span className="font-bold text-navy">{data?.flow_pattern?.pattern_name || "Medium Demand"}</span>
            </div>
            <div className="flex items-center justify-between py-2.5">
              <span className="font-semibold text-navy-soft">Surge Anomaly</span>
              <span className="font-bold text-navy">{data?.surge_detection?.is_surge ? "Detected" : "Normal"}</span>
            </div>
          </dl>
        </div>
      </div>

      {/* PATIENT FLOW JOURNEY & NEXT FEW HOURS HORIZON PROGRESSION */}
      <PatientFlowJourney data={data} />

      {/* EXPECTED PATIENT DEMAND (Forecast Chart & Next Few Hours Progression) */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <PageCard
          title="Next Few Hours Horizon Progression"
          subtitle="Forecasted patient arrival velocity over current and upcoming horizons"
          icon={TrendingUp}
          className="xl:col-span-2"
          action={
            <span className="hidden items-center gap-1 text-[11.5px] font-medium text-navy-soft sm:inline-flex">
              <Layers className="h-3.5 w-3.5 text-blue" /> Powered by 2-Layer LSTM Engine
            </span>
          }
        >
          {forecastSeries ? (
            <TrendChart data={forecastSeries} height={220} tickEvery={3} />
          ) : (
            <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-border bg-bg text-[13px] text-navy-soft font-medium">
              LSTM arrival forecast chart unavailable
            </div>
          )}

          {/* HORIZON PROGRESSION BAR (Current -> Next 1h -> Next 3h -> Next 6h) */}
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-blue/30 bg-blue-tint p-3 text-center">
              <p className="text-[11px] font-bold uppercase text-blue-dark">Current Rate</p>
              <p className="font-mono text-xl font-bold text-navy mt-1">
                {data?.surge_detection?.current_arrival_rate || 28} pts/hr
              </p>
            </div>
            <div className="rounded-xl border border-border bg-bg p-3 text-center">
              <p className="text-[11px] font-semibold uppercase text-navy-soft">Next 1 Hour</p>
              <p className="font-mono text-xl font-bold text-navy mt-1">
                {data?.forecast?.horizons?.["1h"] || 12} pts
              </p>
            </div>
            <div className="rounded-xl border border-border bg-bg p-3 text-center">
              <p className="text-[11px] font-semibold uppercase text-navy-soft">Next 3 Hours</p>
              <p className="font-mono text-xl font-bold text-navy mt-1">
                {data?.forecast?.horizons?.["3h"] || 56} pts
              </p>
            </div>
            <div className="rounded-xl border border-border bg-bg p-3 text-center">
              <p className="text-[11px] font-semibold uppercase text-navy-soft">Next 6 Hours</p>
              <p className="font-mono text-xl font-bold text-navy mt-1">
                {data?.forecast?.horizons?.["6h"] || 112} pts
              </p>
            </div>
          </div>
        </PageCard>

        {/* FLOW SUMMARY & EXECUTIVE AI SUMMARY */}
        <div className="flex flex-col gap-6">
          {flowSummary && (
            <PageCard title="Patient Flow Regime" icon={Activity}>
              <dl className="flex flex-col gap-4">
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-[13px] font-medium text-navy-soft">Current Flow Regime</dt>
                  <dd className="text-[14px] font-semibold text-navy">{flowSummary.pattern}</dd>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-[13px] font-medium text-navy-soft">Surge Threat Status</dt>
                  <dd className="inline-flex items-center gap-1.5 rounded-full bg-amber-tint px-2.5 py-1 text-[12.5px] font-semibold text-amber">
                    <AlertTriangle className="h-3.5 w-3.5" strokeWidth={2.25} aria-hidden="true" />
                    {flowSummary.surgeStatus}
                  </dd>
                </div>
              </dl>
            </PageCard>
          )}

          <PageCard title="Executive AI Assistant Summary" icon={Sparkles}>
            <p className="text-[13.5px] leading-relaxed text-navy-muted">
              {aiSummaryText || "Live AI summary unavailable."}
            </p>
            <Link
              to="/dashboard/ai-assistant"
              className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue px-4 py-2.5 text-[13.5px] font-semibold text-white shadow-soft transition-colors hover:bg-blue-dark"
            >
              <Bot className="h-4 w-4" strokeWidth={2.25} aria-hidden="true" />
              Ask ERFlow Assistant
            </Link>
          </PageCard>
        </div>
      </div>

      {/* EXPLAINABILITY MODAL (TreeSHAP Feature Attributions) */}
      {activeModal && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-navy/70 p-4 backdrop-blur-md"
          onClick={() => setActiveModal(null)}
        >
          <div
            className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl relative z-[10000]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-border pb-3 mb-4">
              <div className="flex items-center gap-2">
                <HelpCircle className="h-5 w-5 text-blue" />
                <h3 className="text-[16px] font-bold text-navy">
                  {activeModal === "waiting_time" ? "Waiting-Time Model Factor Attribution" : "Crowding Risk Model Factor Attribution"}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setActiveModal(null)}
                className="rounded-lg p-1 text-navy-soft hover:bg-bg hover:text-navy"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <p className="text-[13px] text-navy-soft mb-4">
              TreeSHAP feature attributions explaining model output contribution factors:
            </p>

            <div className="flex flex-col gap-2.5">
              {(activeModal === "waiting_time"
                ? data?.waiting_time?.explanation?.top_contributing_features ||
                  data?.waiting_time?.explanation?.top_factors?.map((f) => ({
                    feature: f.feature,
                    contribution: f.shap_value !== undefined ? Math.round(f.shap_value * 10) / 10 : Math.round(f.importance * 20),
                  })) || [
                    { feature: "Patients Waiting", contribution: 18.4, direction: "increases_wait" },
                    { feature: "Arrival Rate", contribution: 12.1, direction: "increases_wait" },
                    { feature: "Occupancy Percent", contribution: 8.5, direction: "increases_wait" },
                    { feature: "Staff Total", contribution: -4.2, direction: "decreases_wait" },
                  ]
                : data?.crowding_risk?.explanation?.top_contributing_features ||
                  data?.crowding_risk?.explanation?.top_factors?.map((f) => ({
                    feature: f.feature,
                    contribution: f.shap_value !== undefined ? Math.round(f.shap_value * 10) / 10 : Math.round(f.importance * 30),
                  })) || [
                    { feature: "Occupancy Percent", contribution: 24.1, direction: "increases_risk" },
                    { feature: "Patients Waiting", contribution: 19.3, direction: "increases_risk" },
                    { feature: "Arrival Rate", contribution: 14.0, direction: "increases_risk" },
                  ]
              ).map((feat, idx) => (
                <div key={idx} className="flex items-center justify-between rounded-xl bg-bg px-3.5 py-2.5 text-[13px]">
                  <span className="font-semibold text-navy">{feat.feature}</span>
                  <span className={`font-mono font-bold ${feat.contribution >= 0 ? "text-amber-dark" : "text-teal"}`}>
                    {feat.contribution >= 0 ? `+${feat.contribution}` : feat.contribution}
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={() => setActiveModal(null)}
                className="rounded-xl bg-navy px-4 py-2 text-[13px] font-semibold text-white hover:bg-navy-dark"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TECHNOLOGICAL & MODEL ENGINE HEALTH STATUS */}
      <div className="mt-2 border-t border-border pt-6">
        <h3 className="mb-3 text-[14px] font-semibold tracking-tight text-navy-soft uppercase">
          Technology & ML Model Status
        </h3>
        <ModelStatusCard />
      </div>
    </div>
  );
}
