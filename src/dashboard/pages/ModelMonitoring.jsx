import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  ShieldAlert,
  Server,
  Zap,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ModelBadge from "../components/ModelBadge";
import StatusBadge from "../components/StatusBadge";
import MetricCard from "../components/MetricCard";
import { erflowApi } from "../../services/api";
import { useMode } from "../../context/ModeContext";

export default function ModelMonitoring() {
  const { isRealMode, isDemoMode } = useMode();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function fetchReport() {
    setLoading(true);
    setError(null);
    try {
      if (isRealMode) {
        const res = await erflowApi.getMonitoringReport();
        setData(res);
      } else {
        // Synthetic Demo Telemetry Data
        setData({
          models: {
            waiting_time_model: {
              model_name: "Supervised XGBoost Regressor",
              status: "online",
              last_latency_ms: 24.2,
              avg_latency_ms: 22.8,
              inference_count: 58,
              error_count: 0,
              validation_failures: 1,
              last_inference_timestamp: new Date().toISOString(),
              drift_info: { drift_status: "Normal", message: "Input features within baseline bounds." },
            },
            crowding_model: {
              model_name: "Supervised XGBoost Classifier",
              status: "online",
              last_latency_ms: 18.5,
              avg_latency_ms: 19.1,
              inference_count: 52,
              error_count: 0,
              validation_failures: 0,
              last_inference_timestamp: new Date().toISOString(),
              drift_info: { drift_status: "Normal", message: "Input features within baseline bounds." },
            },
            high_demand_model: {
              model_name: "Operational Surge Anomaly Detector",
              status: "online",
              last_latency_ms: 12.0,
              avg_latency_ms: 13.4,
              inference_count: 45,
              error_count: 0,
              validation_failures: 0,
              last_inference_timestamp: new Date().toISOString(),
              drift_info: { drift_status: "Normal", message: "Input features within baseline bounds." },
            },
            flow_pattern_model: {
              model_name: "Unsupervised K-Means + PCA",
              status: "online",
              last_latency_ms: 15.3,
              avg_latency_ms: 16.0,
              inference_count: 48,
              error_count: 0,
              validation_failures: 0,
              last_inference_timestamp: new Date().toISOString(),
              drift_info: { drift_status: "Monitoring baseline unavailable", message: "Insufficient inference history (<5 samples)." },
            },
            patient_volume_model: {
              model_name: "Deep Learning LSTM",
              status: "online",
              last_latency_ms: 45.1,
              avg_latency_ms: 48.6,
              inference_count: 36,
              error_count: 0,
              validation_failures: 0,
              last_inference_timestamp: new Date().toISOString(),
              drift_info: { drift_status: "Normal", message: "Input features within baseline bounds." },
            },
          },
          alerts: [],
        });
      }
    } catch (err) {
      console.warn("Monitoring telemetry fetch failed:", err.message);
      setError("Monitoring service offline — unable to retrieve telemetry payload.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchReport();
    const interval = setInterval(fetchReport, 5000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRealMode]);

  const models = data?.models ? Object.values(data.models) : [];
  const alerts = data?.alerts || [];

  const onlineCount = models.filter((m) => m.status === "online").length;
  const avgLatency = models.length > 0
    ? Math.round(models.reduce((acc, m) => acc + (m.avg_latency_ms || m.last_latency_ms || 0), 0) / models.length)
    : 0;
  const totalCalls = models.reduce((acc, m) => acc + (m.inference_count || 0), 0);
  const totalErrors = models.reduce((acc, m) => acc + (m.error_count || 0), 0);

  return (
    <div className="flex flex-col gap-6">
      {/* Demo Mode Notice */}
      {isDemoMode && (
        <div className="flex items-center justify-between rounded-xl border border-amber/40 bg-amber-tint px-4 py-3 text-[13px] text-amber-dark">
          <div className="flex items-center gap-2 font-medium">
            <span className="rounded bg-amber px-2 py-0.5 text-[11px] font-bold text-white uppercase">DEMO MODE</span>
            <span>Displaying synthetic telemetry metrics. Switch to REAL ML MODE in the header for live backend model monitoring.</span>
          </div>
        </div>
      )}

      <PageHeader
        title="ML Model Monitoring & Health"
        subtitle="Real-time inference telemetry, latency performance, error tracking, and input drift monitoring."
        action={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={fetchReport}
              className="flex items-center gap-1.5 rounded-xl border border-border bg-surface px-3 py-2 text-[12.5px] font-semibold text-navy hover:bg-bg"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
            </button>
            <ModelBadge model="Telemetry Layer v1.0" />
          </div>
        }
      />

      {isRealMode && error && (
        <div className="flex items-center justify-between rounded-xl border border-red/30 bg-red-tint px-4 py-3 text-[13px] text-red">
          <div className="flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4 text-red shrink-0" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* SUMMARY METRICS PILLARS */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Active Model Status"
          value={`${onlineCount} / ${models.length || 5} Online`}
          change={onlineCount === 5 ? "100% Operational" : "Degraded State"}
          icon={Server}
          tone={onlineCount === 5 ? "teal" : "amber"}
        />
        <MetricCard
          title="Average Latency"
          value={`${avgLatency} ms`}
          change="Model execution time"
          icon={Zap}
          tone="blue"
        />
        <MetricCard
          title="Total Inferences"
          value={totalCalls.toLocaleString()}
          change="Across all endpoints"
          icon={Activity}
          tone="teal"
        />
        <MetricCard
          title="Inference Errors"
          value={totalErrors.toString()}
          change={totalErrors === 0 ? "0.0% Error rate" : "Failures logged"}
          icon={ShieldAlert}
          tone={totalErrors === 0 ? "teal" : "red"}
        />
      </div>

      {/* SYSTEM ALERTS BANNER */}
      {alerts.length > 0 ? (
        <div className="rounded-2xl border border-amber/40 bg-amber-tint p-5">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="h-5 w-5 text-amber-dark" />
            <h3 className="text-[14px] font-bold text-amber-dark uppercase">Active Operational System Alerts</h3>
          </div>
          <div className="flex flex-col gap-2">
            {alerts.map((alert, idx) => (
              <div key={idx} className="flex items-center justify-between rounded-xl bg-surface px-4 py-2.5 text-[13px] border border-border">
                <span className="font-semibold text-navy">{alert.model}: {alert.message}</span>
                <StatusBadge label={alert.severity.toUpperCase()} tone={alert.severity === "critical" ? "red" : "amber"} />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2 rounded-xl border border-teal/30 bg-teal-tint px-4 py-3 text-[13px] text-teal font-semibold">
          <CheckCircle2 className="h-4 w-4" />
          <span>All 5 ML prediction models are operating cleanly within normal performance parameters.</span>
        </div>
      )}

      {/* DETAILED MODEL TELEMETRY TABLE */}
      <div className="rounded-2xl border border-border bg-surface p-6 shadow-soft">
        <div className="flex items-center justify-between border-b border-border pb-4 mb-4">
          <div>
            <span className="text-[11.5px] font-bold tracking-wider text-navy-soft uppercase">
              Registered ML Artifact Telemetry
            </span>
            <h2 className="text-xl font-bold tracking-tight text-navy">Model Operational Health</h2>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-border text-navy-soft text-[11.5px] font-bold uppercase tracking-wider">
                <th className="py-3 px-3">Model Name</th>
                <th className="py-3 px-3">Status</th>
                <th className="py-3 px-3">Latency</th>
                <th className="py-3 px-3">Inferences</th>
                <th className="py-3 px-3">Errors</th>
                <th className="py-3 px-3">Drift Monitor</th>
                <th className="py-3 px-3">Last Execution</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {models.map((m, idx) => {
                const isOnline = m.status === "online";
                const driftText = m.drift_info?.drift_status || "Monitoring baseline unavailable";
                const hasDrift = m.drift_info?.has_drift;
                return (
                  <tr key={idx} className="hover:bg-bg/50 transition-colors">
                    <td className="py-3.5 px-3">
                      <p className="font-bold text-navy">{m.model_name}</p>
                      <p className="font-mono text-[11px] text-navy-muted">{m.model_key}</p>
                    </td>
                    <td className="py-3.5 px-3">
                      <span className={`inline-flex items-center gap-1.5 font-bold ${isOnline ? "text-teal" : "text-red"}`}>
                        <span className={`h-2 w-2 rounded-full ${isOnline ? "bg-teal" : "bg-red"}`} />
                        {isOnline ? "Online" : "Offline"}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono font-bold text-navy">
                      {m.last_latency_ms ? `${m.last_latency_ms} ms` : "—"}
                    </td>
                    <td className="py-3.5 px-3 font-mono text-navy font-semibold">
                      {m.inference_count || 0}
                    </td>
                    <td className="py-3.5 px-3 font-mono font-semibold">
                      <span className={m.error_count > 0 ? "text-red font-bold" : "text-navy-muted"}>
                        {m.error_count || 0}
                      </span>
                    </td>
                    <td className="py-3.5 px-3">
                      <span
                        className={`text-[12px] font-semibold ${
                          hasDrift ? "text-amber-dark" : driftText === "Normal" ? "text-teal" : "text-navy-muted"
                        }`}
                      >
                        {driftText}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 font-mono text-[11.5px] text-navy-soft">
                      {m.last_inference_timestamp
                        ? new Date(m.last_inference_timestamp).toLocaleTimeString()
                        : "No recent calls"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
