import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  ArrowDownRight,
  CheckCircle2,
  Clock,
  Cpu,
  Flame,
  Layers,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  Users,
  Zap,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import MetricCard from "../components/MetricCard";
import StatusBadge, { LEVEL_TONE } from "../components/StatusBadge";
import ModelBadge from "../components/ModelBadge";
import StepperControl from "../components/StepperControl";
import { erflowApi } from "../../services/api";
import { useMode } from "../../context/ModeContext";

const PRESET_SCENARIOS = {
  quiet: {
    label: "Quiet Shift",
    tone: "green",
    description: "Low patient arrivals, high bed availability, and light triage queue.",
    state: {
      arrival_rate: 10,
      patients_waiting: 5,
      occupancy_percent: 25,
      available_beds: 15,
      available_doctors: 8,
      available_nurses: 12,
      severity_level: 2.0,
      hour_of_day: 3,
      day_of_week: 2,
      month: 7,
    },
  },
  busy: {
    label: "Busy Evening",
    tone: "amber",
    description: "Moderate patient volume with peak evening arrival velocity.",
    state: {
      arrival_rate: 28,
      patients_waiting: 25,
      occupancy_percent: 78,
      available_beds: 8,
      available_doctors: 5,
      available_nurses: 9,
      severity_level: 3.0,
      hour_of_day: 18,
      day_of_week: 4,
      month: 7,
    },
  },
  surge: {
    label: "Surge Scenario",
    tone: "red",
    description: "Severe volume influx, near-capacity bed occupancy, and elevated waiting queue.",
    state: {
      arrival_rate: 52,
      patients_waiting: 58,
      occupancy_percent: 96,
      available_beds: 2,
      available_doctors: 3,
      available_nurses: 5,
      severity_level: 4.2,
      hour_of_day: 20,
      day_of_week: 5,
      month: 7,
    },
  },
};

import CentralContextBanner from "../components/CentralContextBanner";
import { useERContext } from "../../context/ERContext";

/**
 * Helper to generate deterministic scenario predictions in DEMO MODE
 * that logically respond to all 8 operational variables.
 */
function computeDemoPredictions(controls) {
  const arr = controls.arrival_rate ?? 28;
  const occ = controls.occupancy_percent ?? 78;
  const wait = controls.patients_waiting ?? 24;
  const beds = controls.available_beds ?? 8;
  const docs = controls.available_doctors ?? 5;
  const nurses = controls.available_nurses ?? 9;
  const acuity = controls.severity_level ?? 3.0;
  const hour = controls.hour_of_day ?? 18;

  const totalStaff = Math.max(1, docs + nurses);
  const ptsPerStaff = wait / totalStaff;
  const ptsPerBed = wait / (Math.max(1, beds) + 1);

  let timeModifier = 0;
  if (hour >= 17 && hour <= 21) {
    timeModifier = 10;
  } else if (hour >= 1 && hour <= 5) {
    timeModifier = -15;
  }

  const rawWaitTime = 12 + wait * 1.2 + arr * 0.5 + ptsPerStaff * 8.0 + ptsPerBed * 6.0 + (acuity - 3.0) * 10.0 + timeModifier;
  const waitTime = Math.max(5, Math.round(rawWaitTime));

  const rawScore = occ * 0.5 + Math.min(40, ptsPerStaff * 12) + (arr / 50) * 20 + (acuity - 3.0) * 5;
  const score = Math.min(100, Math.max(10, Math.round(rawScore)));

  const level = score >= 85 ? "CRITICAL" : score >= 70 ? "HIGH" : score >= 45 ? "MODERATE" : "LOW";
  const isSurge = arr > 40 || occ > 88 || ptsPerStaff > 4.0;

  let pattern_name = "Medium Demand";
  let cluster_id = 1;
  if (arr >= 38 || occ >= 85 || wait >= 35) {
    pattern_name = "High Demand";
    cluster_id = 0;
  } else if (arr <= 15 && occ <= 50 && wait <= 10) {
    pattern_name = "Low Demand";
    cluster_id = 2;
  }

  const top_factors = [
    { feature: "Patients Waiting", direction: wait > 20 ? "increases" : "decreases", importance: 0.45 },
    { feature: "Occupancy Rate", direction: occ > 75 ? "increases" : "decreases", importance: 0.25 },
    { feature: "Staffing Load", direction: ptsPerStaff > 2.5 ? "increases" : "decreases", importance: 0.18 },
    { feature: "Arrival Velocity", direction: arr > 30 ? "increases" : "decreases", importance: 0.12 },
  ];

  return {
    waiting_time: {
      waiting_time_minutes: waitTime,
      trend: arr > 30 || wait > 25 ? "Increasing" : "Stable",
      explanation: { top_factors },
    },
    crowding_risk: {
      crowding_level: level,
      crowding_score: score,
      explanation: { top_factors },
    },
    flow_pattern: {
      pattern_name,
      cluster_id,
    },
    surge_detection: {
      status: isSurge ? "ANOMALOUS SURGE DETECTED" : "NORMAL OPERATIONAL LOAD",
      is_surge: isSurge,
      severity: isSurge ? (arr > 50 || occ > 90 ? "High" : "Moderate") : "Low",
    },
    patient_forecast: { horizons: { "3h": Math.round(arr * 2.2) } },
  };
}

export default function ScenarioSimulator() {
  const { isRealMode, isDemoMode } = useMode();
  const { predictions, operationalState, setOperationalState } = useERContext();

  const [baselineData, setBaselineData] = useState(predictions);
  const [scenarioData, setScenarioData] = useState(null);
  const [loadingScenario, setLoadingScenario] = useState(false);
  const [error, setError] = useState(null);
  const [activePreset, setActivePreset] = useState("custom");

  // Form Controls State initialized to Central ER State Baseline
  const [scenarioControls, setScenarioControls] = useState(operationalState);

  // Sync baselineData directly with live ERContext predictions & operationalState
  useEffect(() => {
    let isMounted = true;
    async function updateBaseline() {
      if (predictions) {
        setBaselineData(predictions);
      } else {
        try {
          if (isRealMode) {
            const res = await erflowApi.getDashboardOverview(operationalState);
            if (isMounted) setBaselineData(res);
          } else {
            const res = computeDemoPredictions(operationalState);
            if (isMounted) setBaselineData(res);
          }
        } catch {
          // If baseline fetch fails, keep baseline null
        }
      }
    }
    updateBaseline();
    return () => { isMounted = false; };
  }, [isRealMode, predictions, operationalState]);

  // Execute Scenario Analysis against FastAPI Backend or Demo Engine
  async function analyzeScenario(controlsToAnalyze = scenarioControls) {
    setLoadingScenario(true);
    setError(null);
    try {
      if (isRealMode) {
        const res = await erflowApi.getDashboardOverview(controlsToAnalyze);
        setScenarioData(res);
      } else {
        const res = computeDemoPredictions(controlsToAnalyze);
        setScenarioData(res);
      }
    } catch (err) {
      console.warn("Scenario analysis fetch failed:", err.message);
      setError("Scenario analysis unavailable — ML service offline.");
    } finally {
      setLoadingScenario(false);
    }
  }

  // Initial scenario run on mount or mode change
  useEffect(() => {
    analyzeScenario(scenarioControls);
  }, [isRealMode]);

  const resetToCentralBaseline = () => {
    setActivePreset("custom");
    setScenarioControls(operationalState);
    analyzeScenario(operationalState);
  };

  const applyAsCentralBaseline = () => {
    setOperationalState(scenarioControls);
  };

  const handlePresetSelect = (presetKey) => {
    setActivePreset(presetKey);
    const preset = PRESET_SCENARIOS[presetKey];
    if (preset) {
      setScenarioControls(preset.state);
      analyzeScenario(preset.state);
    }
  };

  const updateControl = (field, val) => {
    setActivePreset("custom");
    setScenarioControls((prev) => {
      const updated = { ...prev, [field]: val };
      analyzeScenario(updated);
      return updated;
    });
  };

  // Extract Comparative Metrics (Baseline vs Scenario)
  const curWait = baselineData?.waiting_time?.waiting_time_minutes !== undefined
    ? Math.round(baselineData.waiting_time.waiting_time_minutes)
    : null;
  const scnWait = scenarioData?.waiting_time?.waiting_time_minutes !== undefined
    ? Math.round(scenarioData.waiting_time.waiting_time_minutes)
    : null;
  const diffWait = curWait !== null && scnWait !== null ? Math.round(scnWait - curWait) : null;

  const curCrowd = baselineData?.crowding_risk?.crowding_level ?? "--";
  const scnCrowd = scenarioData?.crowding_risk?.crowding_level ?? "--";

  const curScore = baselineData?.crowding_risk?.crowding_score ?? null;
  const scnScore = scenarioData?.crowding_risk?.crowding_score ?? null;
  const diffScore = curScore !== null && scnScore !== null ? scnScore - curScore : null;

  const curFlow = baselineData?.flow_pattern?.pattern_name ?? "--";
  const scnFlow = scenarioData?.flow_pattern?.pattern_name ?? "--";

  const curSurge = baselineData?.surge_detection?.status ?? "--";
  const scnSurge = scenarioData?.surge_detection?.status ?? "--";

  const curArr = operationalState?.arrival_rate ?? 28;
  const scnArr = scenarioControls.arrival_rate;
  const diffArr = scnArr - curArr;

  const topFactors = scenarioData?.waiting_time?.explanation?.top_factors || scenarioData?.crowding_risk?.explanation?.top_factors || [];

  return (
    <div className="flex flex-col gap-6">
      {/* Demo Mode Notice */}
      {isDemoMode && (
        <div className="flex items-center justify-between rounded-xl border border-amber/40 bg-amber-tint px-4 py-3 text-[13px] text-amber-dark">
          <div className="flex items-center gap-2 font-medium">
            <span className="rounded bg-amber px-2 py-0.5 text-[11px] font-bold text-white uppercase">DEMO MODE</span>
            <span>Displaying synthetic scenario simulations. Switch to REAL ML MODE in the header for live backend model evaluation.</span>
          </div>
        </div>
      )}

      <PageHeader
        title="ER Scenario Simulator"
        subtitle="Modify operational conditions to evaluate how trained ML models respond to capacity changes."
        action={<ModelBadge model="Multi-Model Scenario Engine" />}
      />

      <CentralContextBanner moduleName="ER Scenario Simulator" />

      {isRealMode && error && (
        <div className="flex items-center justify-between rounded-xl border border-red/30 bg-red-tint px-4 py-3 text-[13px] text-red">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4 text-red shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => analyzeScenario()}
            className="flex items-center gap-1 font-semibold underline hover:text-red-dark"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </button>
        </div>
      )}

      {/* PRESET SCENARIOS TOOLBAR */}
      <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-border pb-3">
          <div>
            <h3 className="text-[14.5px] font-bold text-navy">Quick Preset Scenarios</h3>
            <p className="text-[12px] text-navy-soft">
              Presets only populate operational inputs — predictions are determined exclusively by the ML backend.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={resetToCentralBaseline}
              className="inline-flex items-center gap-1.5 rounded-xl border border-blue/40 bg-blue-tint px-3.5 py-2 text-[12.5px] font-bold text-blue hover:bg-blue hover:text-white transition-all shadow-sm"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset to Current ER Conditions</span>
            </button>
            {Object.keys(PRESET_SCENARIOS).map((key) => {
              const p = PRESET_SCENARIOS[key];
              const isActive = activePreset === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => handlePresetSelect(key)}
                  className={`rounded-xl border px-3.5 py-2 text-[12.5px] font-semibold transition-all ${
                    isActive
                      ? "border-blue bg-blue-tint text-blue shadow-soft"
                      : "border-border bg-bg text-navy-soft hover:bg-surface"
                  }`}
                >
                  {p.label}
                </button>
              );
            })}
          </div>
        </div>
        <p className="mt-2.5 text-[12px] italic text-navy-muted">
          💡 Selected: <span className="font-semibold text-navy">{PRESET_SCENARIOS[activePreset]?.label}</span> —{" "}
          {PRESET_SCENARIOS[activePreset]?.description}
        </p>
      </div>

      {/* TWO COLUMN WORKSPACE: CURRENT ER STATE VS SCENARIO CONTROLS */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* LEFT COLUMN: CURRENT ER BASELINE STATE (4 cols) */}
        <div className="flex flex-col gap-4 lg:col-span-4">
          <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <span className="text-[11.5px] font-bold tracking-wider text-navy-soft uppercase">
                Current ER State (Baseline)
              </span>
              <StatusBadge label="Live Baseline" tone="teal" />
            </div>

            <div className="mt-4 flex flex-col gap-3.5">
              <div className="rounded-xl border border-border bg-bg p-3.5">
                <p className="text-[11.5px] font-semibold text-navy-soft">Expected Waiting Time</p>
                <p className="mt-1 font-mono text-2xl font-bold text-navy">
                  {curWait !== null ? `${curWait} min` : "Unavailable"}
                </p>
              </div>

              <div className="rounded-xl border border-border bg-bg p-3.5">
                <p className="text-[11.5px] font-semibold text-navy-soft">Crowding Risk</p>
                <p className="mt-1 font-mono text-2xl font-bold text-navy">{curCrowd}</p>
              </div>

              <div className="rounded-xl border border-border bg-bg p-3.5">
                <p className="text-[11.5px] font-semibold text-navy-soft">Patient Arrival Rate</p>
                <p className="mt-1 font-mono text-2xl font-bold text-navy">{curArr} pts/hr</p>
              </div>

              <div className="rounded-xl border border-border bg-bg p-3.5">
                <p className="text-[11.5px] font-semibold text-navy-soft">Flow Pattern</p>
                <p className="mt-1 font-mono text-lg font-bold text-navy">{curFlow}</p>
              </div>

              <div className="rounded-xl border border-border bg-bg p-3.5">
                <p className="text-[11.5px] font-semibold text-navy-soft">Surge Status</p>
                <p className="mt-1 text-[13px] font-bold text-navy">{curSurge}</p>
              </div>

              <button
                type="button"
                onClick={applyAsCentralBaseline}
                className="mt-1 flex w-full items-center justify-center gap-1.5 rounded-xl border border-teal/40 bg-teal-tint px-3.5 py-2.5 text-[12.5px] font-bold text-teal-dark hover:bg-teal hover:text-white transition-all shadow-sm"
              >
                <CheckCircle2 className="h-4 w-4" />
                <span>Apply Scenario as Central Baseline</span>
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: SCENARIO OPERATIONAL CONTROLS (8 cols) */}
        <div className="flex flex-col gap-4 lg:col-span-8">
          <div className="rounded-2xl border border-border bg-surface p-5 shadow-soft sm:p-6">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div>
                <h3 className="text-[15px] font-bold text-navy">Modify Scenario Controls</h3>
                <p className="text-[12px] text-navy-soft">
                  Adjust operational variables to evaluate hypothetical hospital strain scenarios.
                </p>
              </div>
              <button
                type="button"
                onClick={() => analyzeScenario()}
                disabled={loadingScenario}
                className="flex items-center gap-1.5 rounded-xl bg-blue px-4 py-2.5 text-[13px] font-bold text-white shadow-soft hover:bg-blue-dark disabled:opacity-50"
              >
                {loadingScenario ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4 fill-white" />
                )}
                Analyze Scenario
              </button>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <StepperControl
                label="Expected Arrivals"
                unit="pts/hr"
                value={scenarioControls.arrival_rate}
                onChange={(v) => updateControl("arrival_rate", v)}
                min={0}
                max={100}
                step={1}
              />
              <StepperControl
                label="Patients Waiting"
                unit="pts"
                value={scenarioControls.patients_waiting}
                onChange={(v) => updateControl("patients_waiting", v)}
                min={0}
                max={150}
                step={1}
              />
              <StepperControl
                label="Occupancy"
                unit="%"
                value={scenarioControls.occupancy_percent}
                onChange={(v) => updateControl("occupancy_percent", v)}
                min={0}
                max={100}
                step={1}
              />
              <StepperControl
                label="Available Beds"
                unit="beds"
                value={scenarioControls.available_beds}
                onChange={(v) => updateControl("available_beds", v)}
                min={0}
                max={50}
                step={1}
              />
              <StepperControl
                label="Available Doctors"
                unit="docs"
                value={scenarioControls.available_doctors}
                onChange={(v) => updateControl("available_doctors", v)}
                min={1}
                max={25}
                step={1}
              />
              <StepperControl
                label="Available Nurses"
                unit="nurses"
                value={scenarioControls.available_nurses}
                onChange={(v) => updateControl("available_nurses", v)}
                min={1}
                max={40}
                step={1}
              />
              <StepperControl
                label="Average Acuity"
                unit="lvl"
                value={scenarioControls.severity_level}
                onChange={(v) => updateControl("severity_level", v)}
                min={1.0}
                max={5.0}
                step={0.1}
              />
              <StepperControl
                label="Hour of Day"
                unit=":00"
                value={scenarioControls.hour_of_day}
                onChange={(v) => updateControl("hour_of_day", v)}
                min={0}
                max={23}
                step={1}
              />
            </div>
          </div>
        </div>
      </div>

      {/* SCENARIO IMPACT COMPARISON RESULT SECTION */}
      {scenarioData && (
        <div className="rounded-2xl border border-border bg-surface p-6 shadow-soft">
          <div className="flex items-center justify-between border-b border-border pb-4">
            <div>
              <span className="text-[11.5px] font-bold tracking-wider text-navy-soft uppercase">
                Model Evaluation Result
              </span>
              <h2 className="text-xl font-bold tracking-tight text-navy">Scenario Impact Summary</h2>
            </div>
            <StatusBadge label="Evaluated via FastAPI Models" tone="blue" />
          </div>

          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {/* 1. Waiting Time Impact */}
            <div className="flex flex-col justify-between rounded-xl border border-border bg-bg p-4">
              <span className="text-[12px] font-semibold text-navy-soft">Waiting Time</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-[13px] text-navy-muted">{curWait}m</span>
                <ArrowRight className="h-3.5 w-3.5 text-navy-soft" />
                <span className="font-mono text-xl font-bold text-navy">{scnWait}m</span>
              </div>
              {diffWait !== null && (
                <div
                  className={`mt-2 flex items-center gap-1 text-[12px] font-bold ${
                    diffWait > 0 ? "text-amber-dark" : diffWait < 0 ? "text-teal" : "text-navy-soft"
                  }`}
                >
                  {diffWait > 0 ? <ArrowUpRight className="h-3.5 w-3.5" /> : <ArrowDownRight className="h-3.5 w-3.5" />}
                  <span>
                    {diffWait > 0 ? `+${diffWait}` : diffWait} min ({diffWait >= 0 ? "+" : ""}
                    {curWait ? Math.round((diffWait / curWait) * 100) : 0}%)
                  </span>
                </div>
              )}
            </div>

            {/* 2. Crowding Impact */}
            <div className="flex flex-col justify-between rounded-xl border border-border bg-bg p-4">
              <span className="text-[12px] font-semibold text-navy-soft">Crowding Risk</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-[13px] text-navy-muted">{curCrowd}</span>
                <ArrowRight className="h-3.5 w-3.5 text-navy-soft" />
                <span className="font-mono text-xl font-bold text-navy">{scnCrowd}</span>
              </div>
              {diffScore !== null && (
                <div
                  className={`mt-2 flex items-center gap-1 text-[12px] font-bold ${
                    diffScore > 0 ? "text-amber-dark" : diffScore < 0 ? "text-teal" : "text-navy-soft"
                  }`}
                >
                  <span>Score: {curScore} → {scnScore} ({diffScore >= 0 ? "+" : ""}{diffScore})</span>
                </div>
              )}
            </div>

            {/* 3. Patient Demand Impact */}
            <div className="flex flex-col justify-between rounded-xl border border-border bg-bg p-4">
              <span className="text-[12px] font-semibold text-navy-soft">Patient Demand</span>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-[13px] text-navy-muted">{curArr}/h</span>
                <ArrowRight className="h-3.5 w-3.5 text-navy-soft" />
                <span className="font-mono text-xl font-bold text-navy">{scnArr}/h</span>
              </div>
              <div
                className={`mt-2 flex items-center gap-1 text-[12px] font-bold ${
                  diffArr > 0 ? "text-amber-dark" : diffArr < 0 ? "text-teal" : "text-navy-soft"
                }`}
              >
                <span>{diffArr >= 0 ? `+${diffArr}` : diffArr} pts/hr</span>
              </div>
            </div>

            {/* 4. Flow Pattern Impact */}
            <div className="flex flex-col justify-between rounded-xl border border-border bg-bg p-4">
              <span className="text-[12px] font-semibold text-navy-soft">Flow Pattern</span>
              <div className="mt-2 flex flex-col gap-1">
                <span className="text-[12px] text-navy-muted">{curFlow}</span>
                <span className="font-mono text-base font-bold text-navy">→ {scnFlow}</span>
              </div>
            </div>

            {/* 5. Surge Anomaly Impact */}
            <div className="flex flex-col justify-between rounded-xl border border-border bg-bg p-4">
              <span className="text-[12px] font-semibold text-navy-soft">Surge Status</span>
              <div className="mt-2 flex flex-col gap-1">
                <span className="text-[12px] text-navy-muted">{curSurge.includes("SURGE") ? "SURGE" : "NORMAL"}</span>
                <span className={`font-mono text-sm font-bold ${scnSurge.includes("SURGE") ? "text-red" : "text-teal"}`}>
                  → {scnSurge}
                </span>
              </div>
            </div>
          </div>

          {/* EXPLAINABLE AI LAYER: WHY DID THE SCENARIO CHANGE? */}
          {topFactors.length > 0 && (
            <div className="mt-6 border-t border-border pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-4 w-4 text-teal" />
                <h4 className="text-[13.5px] font-bold text-navy">Why Did the Scenario Change? (TreeSHAP Model Factors)</h4>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {topFactors.map((factor, idx) => {
                  const isIncrease = factor.direction === "increases";
                  return (
                    <div key={idx} className="flex items-center justify-between rounded-xl border border-border bg-bg px-4 py-2.5 text-[12.5px]">
                      <span className="font-semibold text-navy">{factor.feature}</span>
                      <span className={`font-mono font-bold ${isIncrease ? "text-amber-dark" : "text-teal"}`}>
                        {isIncrease ? "↑" : "↓"} {Math.round(factor.importance * 100)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
