import React, { useState, useEffect } from "react";
import { Sliders, RefreshCw, CheckCircle2, Play, Activity, Cpu } from "lucide-react";
import StepperControl from "./StepperControl";
import { useERContext } from "../../context/ERContext";
export default function EROperationsControlPanel({ className = "" }) {
  const {
    operationalState,
    setOperationalState,
    updatePredictions,
    loading,
    error,
    lastUpdated,
    modelStatus,
  } = useERContext();

  const [form, setForm] = useState(operationalState);

  useEffect(() => {
    setForm(operationalState);
  }, [operationalState]);

  const handleChange = (field, val) => {
    const updated = { ...form, [field]: val };
    setForm(updated);
    setOperationalState(updated);
    updatePredictions(updated);
  };

  const handleUpdateAll = async (e) => {
    e.preventDefault();
    await updatePredictions(form);
  };

  return (
    <div className={`rounded-2xl border border-border bg-surface p-4 shadow-soft sm:p-5 ${className}`}>
      {/* HEADER */}
      <div className="flex flex-col gap-2 border-b border-border pb-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-tint text-blue">
            <Sliders className="h-4 w-4" strokeWidth={2.25} />
          </span>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-[15px] font-bold text-navy">ER Operations Control Panel</h2>
              <span className="rounded-full bg-blue-tint px-2 py-0.5 text-[10px] font-bold text-blue uppercase">
                Centralized Command
              </span>
            </div>
            <p className="mt-0.5 text-[12px] text-navy-soft">
              Update central ER operational variables to synchronize all 5 ML prediction engines
            </p>
          </div>
        </div>

        {lastUpdated && (
          <div className="flex items-center gap-1.5 rounded-lg border border-green/30 bg-green-tint/40 px-2.5 py-1 text-[11.5px] font-semibold text-green shrink-0">
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-green" />
            <span>Synced at {lastUpdated}</span>
          </div>
        )}
      </div>

      {/* FORM INPUTS GRID */}
      <form onSubmit={handleUpdateAll} className="mt-3.5 flex flex-col gap-3.5">
        <div className="grid grid-cols-1 gap-3.5 lg:grid-cols-2">
          {/* SECTION 1: CURRENT ER STATUS */}
          <div className="rounded-xl border border-border/60 bg-bg/40 p-3">
            <div className="mb-2.5 flex items-center gap-1.5 border-b border-border/40 pb-1.5">
              <Activity className="h-3.5 w-3.5 text-blue" />
              <h3 className="text-[11.5px] font-bold uppercase tracking-wider text-navy">
                1. Current ER Operational Status
              </h3>
            </div>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
              <StepperControl
                label="ER Occupancy"
                value={form.occupancy_percent}
                onChange={(val) => handleChange("occupancy_percent", val)}
                min={20}
                max={100}
                step={1}
                unit="%"
              />
              <StepperControl
                label="Patients Waiting"
                value={form.patients_waiting}
                onChange={(val) => handleChange("patients_waiting", val)}
                min={0}
                max={80}
                step={1}
                unit="pts"
              />
              <StepperControl
                label="Available Beds"
                value={form.available_beds}
                onChange={(val) => handleChange("available_beds", val)}
                min={0}
                max={40}
                step={1}
                unit="beds"
              />
            </div>
          </div>

          {/* SECTION 2: DEMAND & STAFFING */}
          <div className="rounded-xl border border-border/60 bg-bg/40 p-3">
            <div className="mb-2.5 flex items-center gap-1.5 border-b border-border/40 pb-1.5">
              <Cpu className="h-3.5 w-3.5 text-teal" />
              <h3 className="text-[11.5px] font-bold uppercase tracking-wider text-navy">
                2. Patient Demand & Staffing Inputs
              </h3>
            </div>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-4">
              <StepperControl
                label="Arrival Velocity"
                value={form.arrival_rate}
                onChange={(val) => handleChange("arrival_rate", val)}
                min={5}
                max={60}
                step={1}
                unit="pts/hr"
              />
              <StepperControl
                label="Severity Acuity"
                value={form.severity_level}
                onChange={(val) => handleChange("severity_level", val)}
                min={1}
                max={5}
                step={0.5}
                unit="level"
              />
              <StepperControl
                label="Active Physicians"
                value={form.available_doctors}
                onChange={(val) => handleChange("available_doctors", val)}
                min={1}
                max={20}
                step={1}
                unit="docs"
              />
              <StepperControl
                label="Active Nurses"
                value={form.available_nurses}
                onChange={(val) => handleChange("available_nurses", val)}
                min={1}
                max={40}
                step={1}
                unit="nurses"
              />
            </div>
          </div>
        </div>

        {/* PRIMARY ACTION BAR */}
        <div className="flex flex-col gap-2.5 sm:flex-row sm:items-center sm:justify-between border-t border-border/80 pt-3">
          <div className="flex flex-wrap items-center gap-1.5 text-[11.5px] font-medium text-navy-soft">
            <span className="font-semibold text-navy">Engine Pipelines:</span>
            <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-bold ${
              modelStatus.forecast === "success" ? "bg-green-tint text-green" : "bg-bg text-navy-soft"
            }`}>
              ✓ Forecast
            </span>
            <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-bold ${
              modelStatus.waiting_time === "success" ? "bg-green-tint text-green" : "bg-bg text-navy-soft"
            }`}>
              ✓ Wait Time
            </span>
            <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-bold ${
              modelStatus.crowding_risk === "success" ? "bg-green-tint text-green" : "bg-bg text-navy-soft"
            }`}>
              ✓ Crowding
            </span>
            <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-bold ${
              modelStatus.flow_pattern === "success" ? "bg-green-tint text-green" : "bg-bg text-navy-soft"
            }`}>
              ✓ Patterns
            </span>
            <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-bold ${
              modelStatus.surge_detection === "success" ? "bg-green-tint text-green" : "bg-bg text-navy-soft"
            }`}>
              ✓ Surge
            </span>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue px-5 py-2 text-[13px] font-bold text-white shadow-soft hover:bg-blue-dark transition-all disabled:opacity-50 shrink-0"
          >
            {loading ? (
              <>
                <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                <span>Updating Predictions...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-white" />
                <span>Update All Predictions</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red/30 bg-red-tint p-2.5 text-[12px] font-semibold text-red">
            {error}
          </div>
        )}
      </form>
    </div>
  );
}
