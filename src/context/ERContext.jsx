import React, { createContext, useContext, useEffect, useState } from "react";
import { erflowApi } from "../services/api";
import { useMode } from "./ModeContext";

const now = new Date();
const currentHour = now.getHours();
const currentDay = now.getDay();
const currentMonth = now.getMonth() + 1;

const DEFAULT_OPERATIONAL_STATE = {
  occupancy_percent: 78,
  patients_waiting: 24,
  arrival_rate: 28,
  available_beds: 8,
  available_doctors: 5,
  available_nurses: 9,
  severity_level: 3.0,
  hour_of_day: currentHour,
  day_of_week: currentDay,
  month: currentMonth,
};

const ERContext = createContext({
  operationalState: DEFAULT_OPERATIONAL_STATE,
  setOperationalState: () => {},
  predictions: null,
  loading: false,
  error: null,
  lastUpdated: null,
  modelStatus: {
    forecast: "idle",
    waiting_time: "idle",
    crowding_risk: "idle",
    flow_pattern: "idle",
    surge_detection: "idle",
  },
  updatePredictions: async () => {},
  resetToBaseline: () => {},
});

export function ERProvider({ children }) {
  const { isRealMode } = useMode();

  const [operationalState, setOperationalStateState] = useState(() => {
    try {
      const saved = localStorage.getItem("erflow_operational_state");
      return saved ? JSON.parse(saved) : DEFAULT_OPERATIONAL_STATE;
    } catch {
      return DEFAULT_OPERATIONAL_STATE;
    }
  });

  const [predictions, setPredictions] = useState(() => {
    try {
      const saved = sessionStorage.getItem("erflow_predictions");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [hasRunPredictions, setHasRunPredictions] = useState(() => {
    try {
      return Boolean(sessionStorage.getItem("erflow_predictions"));
    } catch {
      return false;
    }
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(() => {
    return localStorage.getItem("erflow_last_updated") || null;
  });

  const [modelStatus, setModelStatus] = useState({
    forecast: "idle",
    waiting_time: "idle",
    crowding_risk: "idle",
    flow_pattern: "idle",
    surge_detection: "idle",
  });

  const setOperationalState = (newState) => {
    setOperationalStateState(newState);
    try {
      localStorage.setItem("erflow_operational_state", JSON.stringify(newState));
    } catch (e) {
      console.warn("Failed to persist operationalState to localStorage:", e);
    }
    updatePredictions(newState);
  };

function generateDemoPredictions(state) {
  const occ = Number(state.occupancy_percent ?? 78);
  const waitPts = Number(state.patients_waiting ?? 24);
  const arrRate = Number(state.arrival_rate ?? 28);
  const availBeds = Number(state.available_beds ?? 8);
  const docs = Number(state.available_doctors ?? 5);

  const isSurge = arrRate > 35 || occ > 85;
  const rawWait = Math.round(10 + waitPts * 1.2 + arrRate * 0.6 - docs * 1.5 - availBeds * 0.5);
  const finalWaitMin = Math.max(5, Math.min(180, rawWait));

  const crowdingScore = Math.min(100, Math.max(0, Math.round(occ * 0.5 + (waitPts / 50) * 30 + (arrRate / 40) * 20)));
  const crowdingLevel = crowdingScore >= 80 ? "CRITICAL" : crowdingScore >= 60 ? "HIGH" : crowdingScore >= 35 ? "MODERATE" : "LOW";

  const patternName = arrRate > 35 ? "High Demand" : arrRate < 18 ? "Low Demand" : "Medium Demand";

  return {
    forecast: {
      predicted_peak_time: "7:00 PM",
      predicted_peak_rate: Math.round(arrRate * 1.2),
      trend: arrRate > 25 ? "Increasing" : "Stable",
      horizons: {
        "1h": Math.round(arrRate * 0.9),
        "3h": Math.round(arrRate * 2.4),
        "6h": Math.round(arrRate * 4.8),
        "24h": Math.round(arrRate * 18.0),
      },
      forecast_cards: [
        { id: "1h", label: "Next 1 Hour", value: `${Math.round(arrRate * 0.9)}`, unit: "arrivals" },
        { id: "3h", label: "Next 3 Hours", value: `${Math.round(arrRate * 2.4)}`, unit: "arrivals" },
        { id: "6h", label: "Next 6 Hours", value: `${Math.round(arrRate * 4.8)}`, unit: "arrivals" },
        { id: "24h", label: "Next 24 Hours", value: `${Math.round(arrRate * 18.0)}`, unit: "arrivals" },
      ],
      series: [
        { time: "12:00 PM", actual: Math.round(arrRate * 0.7), forecast: null },
        { time: "1:00 PM", actual: Math.round(arrRate * 0.8), forecast: null },
        { time: "2:00 PM", actual: Math.round(arrRate * 0.9), forecast: null },
        { time: "3:00 PM", actual: Math.round(arrRate * 1.0), forecast: null },
        { time: "4:00 PM", actual: Math.round(arrRate * 1.1), forecast: null },
        { time: "5:00 PM", actual: Math.round(arrRate * 1.15), forecast: Math.round(arrRate * 1.15) },
        { time: "6:00 PM", actual: null, forecast: Math.round(arrRate * 1.2) },
        { time: "7:00 PM", actual: null, forecast: Math.round(arrRate * 1.25) },
        { time: "8:00 PM", actual: null, forecast: Math.round(arrRate * 1.1) },
        { time: "9:00 PM", actual: null, forecast: Math.round(arrRate * 0.95) },
      ],
      model_name: "2-Layer LSTM Neural Network",
      data_source: "Synthetic Demo Forecast Engine",
      validation_metrics: { mae: 4.42, rmse: 5.81 },
    },
    waiting_time: {
      waiting_time_minutes: finalWaitMin,
      predicted_1h: Math.round(finalWaitMin * 1.1),
      predicted_peak: Math.round(finalWaitMin * 1.35),
      trend: arrRate > 25 ? "Increasing" : "Stable",
      model_name: "Supervised XGBoost Regressor",
      hourly_trend: [
        { t: "12 PM", value: Math.round(finalWaitMin * 0.8), kind: "observed" },
        { t: "3 PM", value: Math.round(finalWaitMin * 0.9), kind: "observed" },
        { t: "6 PM", value: finalWaitMin, kind: "observed" },
        { t: "9 PM (proj.)", value: Math.round(finalWaitMin * 1.2), kind: "forecast" },
      ],
      explanation: {
        top_factors: [
          { feature: "Patients Waiting", direction: "increases", importance: 0.45 },
          { feature: "Arrival Velocity", direction: "increases", importance: 0.30 },
          { feature: "Available Doctors", direction: "decreases", importance: 0.15 },
          { feature: "Available Beds", direction: "decreases", importance: 0.10 },
        ],
        top_contributing_features: [
          { feature: "Patients Waiting", contribution: 18.4, direction: "increases_wait" },
          { feature: "Arrival Rate", contribution: 12.1, direction: "increases_wait" },
          { feature: "Occupancy Percent", contribution: 8.5, direction: "increases_wait" },
          { feature: "Staff Total", contribution: -4.2, direction: "decreases_wait" },
        ],
      },
    },
    crowding_risk: {
      crowding_level: crowdingLevel,
      crowding_score: crowdingScore,
      expected_window: "Next 3 Hours",
      class_probability: crowdingScore / 100,
      probabilities: {
        Critical: crowdingLevel === "CRITICAL" ? 0.75 : 0.1,
        High: crowdingLevel === "HIGH" ? 0.70 : 0.15,
        Moderate: crowdingLevel === "MODERATE" ? 0.65 : 0.15,
        Low: crowdingLevel === "LOW" ? 0.80 : 0.05,
      },
      model_name: "Supervised XGBoost Classifier",
      risk_timeline: [
        { time: "3 PM", level: "MODERATE" },
        { time: "5 PM", level: crowdingLevel === "CRITICAL" ? "HIGH" : crowdingLevel },
        { time: "7 PM", level: crowdingLevel },
        { time: "9 PM", level: crowdingLevel === "CRITICAL" ? "HIGH" : "MODERATE" },
      ],
      explanation: {
        top_factors: [
          { feature: "Occupancy Percent", direction: "increases", importance: 0.50 },
          { feature: "Patients Waiting", direction: "increases", importance: 0.30 },
          { feature: "Arrival Velocity", direction: "increases", importance: 0.20 },
        ],
        top_contributing_features: [
          { feature: "Occupancy Percent", contribution: 24.1, direction: "increases_risk" },
          { feature: "Patients Waiting", contribution: 19.3, direction: "increases_risk" },
          { feature: "Arrival Rate", contribution: 14.0, direction: "increases_risk" },
        ],
      },
    },
    flow_pattern: {
      pattern_name: patternName,
      confidence: 91.5,
      cluster_id: patternName === "High Demand" ? 0 : patternName === "Low Demand" ? 2 : 1,
      description: `Current ER demand exhibits a ${patternName} regime with expected throughput matching operational inputs.`,
      current_point: { x: arrRate / 10, y: occ / 50 },
      model_name: "Unsupervised K-Means + PCA",
    },
    surge_detection: {
      status: isSurge ? "ANOMALOUS SURGE DETECTED" : "NORMAL OPERATIONAL LOAD",
      is_surge: isSurge,
      severity: isSurge ? (arrRate > 45 ? "High" : "Moderate") : "Low",
      normal_arrival_rate: 21,
      current_arrival_rate: arrRate,
      deviation_percent: `${Math.round(((arrRate - 21) / 21) * 100)}%`,
      detected_at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      description: isSurge
        ? `Surge anomaly detected: Current arrival velocity (${arrRate} pts/hr) exceeds baseline (21 pts/hr).`
        : `Arrival velocity (${arrRate} pts/hr) is within normal statistical parameters.`,
      model_name: "Unsupervised DBSCAN Anomaly Engine",
      timeline: [
        { t: "3 PM", expected: 21, actual: Math.round(arrRate * 0.7), anomaly: false },
        { t: "4 PM", expected: 21, actual: Math.round(arrRate * 0.85), anomaly: false },
        { t: "5 PM", expected: 21, actual: arrRate, anomaly: isSurge },
        { t: "6 PM", expected: 21, actual: Math.round(arrRate * 1.1), anomaly: isSurge },
      ],
    },
    ai_summary_text: `Patient demand is projected at ${Math.round(arrRate * 2.4)} arrivals over the next 3 hours. Expected wait time is approximately ${finalWaitMin} minutes with a ${crowdingLevel} crowding risk (score: ${crowdingScore}/100). ER flow pattern exhibits ${patternName} conditions with ${isSurge ? "an abnormal arrival surge" : "normal operational load"}.`,
  };
}

  async function updatePredictions(customState = null) {
    const stateToUse = customState || operationalState;
    setLoading(true);
    setError(null);
    setModelStatus({
      forecast: "updating",
      waiting_time: "updating",
      crowding_risk: "updating",
      flow_pattern: "updating",
      surge_detection: "updating",
    });

    try {
      const res = isRealMode
        ? await erflowApi.getDashboardOverview(stateToUse)
        : generateDemoPredictions(stateToUse);

      if (res) {
        setPredictions(res);
        setHasRunPredictions(true);
        const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        setLastUpdated(now);
        setModelStatus({
          forecast: isRealMode ? "success" : "demo",
          waiting_time: isRealMode ? "success" : "demo",
          crowding_risk: isRealMode ? "success" : "demo",
          flow_pattern: isRealMode ? "success" : "demo",
          surge_detection: isRealMode ? "success" : "demo",
        });

        try {
          sessionStorage.setItem("erflow_predictions", JSON.stringify(res));
          sessionStorage.setItem("erflow_last_updated", now);
        } catch (e) {
          console.warn("Failed to persist predictions:", e);
        }
      }
    } catch (err) {
      console.warn("Central prediction update failed:", err.message);
      setError(err.message || "Failed to update multi-model predictions.");
      setModelStatus({
        forecast: "error",
        waiting_time: "error",
        crowding_risk: "error",
        flow_pattern: "error",
        surge_detection: "error",
      });
    } finally {
      setLoading(false);
    }
  }

  const resetToBaseline = () => {
    setOperationalState(DEFAULT_OPERATIONAL_STATE);
  };

  useEffect(() => {
    if (!predictions) {
      updatePredictions(operationalState);
    }
  }, [isRealMode]);

  const value = {
    operationalState,
    setOperationalState,
    predictions,
    hasRunPredictions,
    loading,
    error,
    lastUpdated,
    modelStatus,
    updatePredictions,
    resetToBaseline,
    defaultOperationalState: DEFAULT_OPERATIONAL_STATE,
  };

  return <ERContext.Provider value={value}>{children}</ERContext.Provider>;
}

export function useERContext() {
  return useContext(ERContext);
}
