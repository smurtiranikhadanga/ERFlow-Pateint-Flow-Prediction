import { Activity, AlertTriangle, PieChart, RefreshCw, ScatterChart } from "lucide-react";
import PageHeader from "../components/PageHeader";
import ChartCard from "../components/ChartCard";
import StatusBadge from "../components/StatusBadge";
import ModelBadge from "../components/ModelBadge";
import MLContextCard from "../components/MLContextCard";
import ClusterScatter from "../components/ClusterScatter";
import BarList from "../components/BarList";
import {
  CURRENT_PATTERN as MOCK_CURRENT,
  FLOW_PATTERN_CARDS as MOCK_CARDS,
  FLOW_PATTERN_DISTRIBUTION as MOCK_DIST,
  FLOW_CLUSTERS as MOCK_CLUSTERS,
  FLOW_CLUSTER_POINTS as MOCK_POINTS,
  FLOW_CURRENT_POINT as MOCK_POINT,
  FLOW_ANALYSIS_MODEL as MOCK_MODEL,
} from "../mockData";
import { useMode } from "../../context/ModeContext";

const MOCK_DISTRIBUTION = MOCK_DIST;

const DOT_TONE = {
  green: "bg-green",
  amber: "bg-amber",
  blue: "bg-blue",
  red: "bg-red",
  teal: "bg-teal",
  navy: "bg-navy",
};

function PatternCard({ name, tone, characteristics, active }) {
  return (
    <div
      className={`rounded-2xl border p-4 shadow-soft ${
        active ? "border-blue/30 bg-blue-tint" : "border-border bg-surface"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <p className={`text-[14.5px] font-semibold ${active ? "text-blue-dark" : "text-navy"}`}>{name}</p>
        {active && <StatusBadge label="ACTIVE" tone="blue" />}
      </div>
      <ul className="mt-3 flex flex-col gap-1.5">
        {characteristics.map((c) => (
          <li key={c} className="flex items-start gap-2 text-[13px] leading-relaxed text-navy-muted">
            <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${DOT_TONE[tone] || DOT_TONE.navy}`} />
            {c}
          </li>
        ))}
      </ul>
    </div>
  );
}

function parseConfidence(val) {
  if (val === null || val === undefined || val === "") return null;
  const num = typeof val === "number" ? val : parseFloat(val);
  if (!Number.isFinite(num)) return null;
  return Math.round(num);
}

function getHumanDescription(patternName, fallbackDesc) {
  const name = (patternName || "").toLowerCase();
  if (name.includes("low")) {
    return "Current ER demand is relatively low.";
  }
  if (name.includes("medium")) {
    return "Current ER demand is at a moderate operational level.";
  }
  if (name.includes("high")) {
    return "Current ER demand is elevated.";
  }
  if (fallbackDesc && !fallbackDesc.includes("K-Means Cluster")) {
    return fallbackDesc;
  }
  return "Current ER demand is at a normal operational baseline.";
}

import CentralContextBanner from "../components/CentralContextBanner";
import { useERContext } from "../../context/ERContext";

export default function FlowPatterns() {
  const { isRealMode, isDemoMode } = useMode();
  const { predictions, operationalState, loading: _loading, error, updatePredictions } = useERContext();

  const data = isRealMode ? predictions?.flow_pattern || null : null;

  const confVal = parseConfidence(data?.confidence);

  const currentPattern = (isRealMode
    ? data
      ? {
          name: data.pattern_name,
          confidence: confVal !== null ? `${confVal}%` : null,
          clusterId: data.cluster_id !== undefined && data.cluster_id !== null ? data.cluster_id : 1,
          description: getHumanDescription(data.pattern_name, data.description),
        }
      : {
          name: "--",
          confidence: "--",
          clusterId: "--",
          description: "Predictions pending. Click 'Update All Predictions' in Overview to run flow pattern clustering.",
        }
    : {
        ...MOCK_CURRENT,
        confidence: parseConfidence(MOCK_CURRENT?.confidence) !== null
          ? `${parseConfidence(MOCK_CURRENT.confidence)}%`
          : null,
        clusterId: 1,
        description: getHumanDescription(MOCK_CURRENT?.name, MOCK_CURRENT?.description),
      });

  const currentPoint = isRealMode
    ? data?.current_point
      ? {
          x: Math.max(5, Math.min(95, Math.round(((data.current_point.x + 3.5) / 9.5) * 100))),
          y: Math.max(5, Math.min(95, Math.round(((data.current_point.y + 1.5) / 3.0) * 100))),
          clusterId: data.cluster_id,
        }
      : null
    : MOCK_POINT;

  // Highlight active pattern in cards array
  const activePattern = (currentPattern?.name || "").toLowerCase();
  const patternCards = MOCK_CARDS.map((card) => {
    const cardName = card.name.toLowerCase();
    const isMatch =
      cardName === activePattern ||
      (activePattern.includes("low") && (cardName.includes("normal") || cardName.includes("low"))) ||
      (activePattern.includes("medium") && (cardName.includes("busy") || cardName.includes("medium"))) ||
      (activePattern.includes("high") && (cardName.includes("extreme") || cardName.includes("peak") || cardName.includes("high")));
    return {
      ...card,
      active: data ? isMatch : card.active,
    };
  });

  const modelName = data?.model_name || MOCK_MODEL;

  return (
    <div className="flex flex-col gap-6">
      {/* Demo Mode Notice */}
      {isDemoMode && (
        <div className="flex items-center justify-between rounded-xl border border-amber/40 bg-amber-tint px-4 py-3 text-[13px] text-amber-dark">
          <div className="flex items-center gap-2 font-medium">
            <span className="rounded bg-amber px-2 py-0.5 text-[11px] font-bold text-white uppercase">DEMO MODE</span>
            <span>Displaying synthetic flow pattern clusters. Switch to REAL ML MODE in the header for live K-Means predictions.</span>
          </div>
        </div>
      )}

      <PageHeader
        title="Patient Flow Pattern Discovery"
        subtitle="Explore recurring emergency department demand patterns identified from historical data."
        action={<ModelBadge model={modelName} />}
      />

      <CentralContextBanner moduleName="Patient Flow Patterns" />

      {isRealMode && error && (
        <div className="flex items-center justify-between rounded-xl border border-red/30 bg-red-tint px-4 py-3 text-[13px] text-red">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4 text-red shrink-0" />
            <span>Prediction Unavailable: Unable to connect to K-Means clustering service.</span>
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

      {/* CONTEXTUAL ML PRESENTATION LAYER */}
      <MLContextCard
        sees={[
          `${operationalState?.arrival_rate || 28} expected arrivals/hr`,
          `${operationalState?.occupancy_percent || 78}% occupancy`,
          `${operationalState?.patients_waiting || 24} patients waiting`,
        ]}
        predicts={`Operational Regime: ${currentPattern.name}`}
        when="Current Shift Window"
        source={modelName}
      />

      <ChartCard title="Current Detected Pattern" icon={Activity}>
        <div>
          <p className="text-[20px] font-semibold text-navy">{currentPattern.name}</p>
          <p className="mt-2 max-w-3xl text-[13.5px] leading-relaxed text-navy-muted">
            {currentPattern.description}
          </p>
        </div>
      </ChartCard>

      <div>
        <h3 className="mb-3 text-[14.5px] font-semibold text-navy">Detected Patterns</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {patternCards.map((p) => (
            <PatternCard key={p.id} {...p} />
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <ChartCard
          title="Patient-Flow Clusters"
          subtitle="Illustrative grouping of ER states by arrival volume and system strain"
          icon={ScatterChart}
          className="xl:col-span-2"
        >
          <ClusterScatter
            clusters={MOCK_CLUSTERS}
            points={MOCK_POINTS}
            currentPoint={currentPoint}
          />
        </ChartCard>

        <ChartCard
          title="Pattern Distribution"
          subtitle="How frequently each pattern occurs"
          icon={PieChart}
        >
          <BarList
            items={MOCK_DISTRIBUTION.map((p) => ({
              label: p.label,
              value: p.value,
              max: 100,
              tone: p.tone,
              valueLabel: `${p.value}%`,
            }))}
          />
        </ChartCard>
      </div>
    </div>
  );
}


