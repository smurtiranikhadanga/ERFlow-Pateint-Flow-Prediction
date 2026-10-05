import React from "react";
import { Link } from "react-router-dom";
import { Activity, Sliders, ArrowRight, CheckCircle2 } from "lucide-react";
import { useERContext } from "../../context/ERContext";

export default function CentralContextBanner({ moduleName: _moduleName = "Current Module" }) {
  const { operationalState, lastUpdated, hasRunPredictions } = useERContext();

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-blue/20 bg-blue-tint/30 p-4 shadow-soft sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue text-white shadow-soft">
          <Activity className="h-4 w-4" />
        </span>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-bold text-navy">
              Centralized Operational Context
            </span>
            <span className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-bold border ${
              hasRunPredictions && lastUpdated
                ? "bg-green-tint text-green border-green/30"
                : "bg-amber-tint text-amber-dark border-amber/30"
            }`}>
              <CheckCircle2 className="h-3 w-3" />
              {hasRunPredictions && lastUpdated ? `Synced ${lastUpdated}` : "Predictions Pending"}
            </span>
          </div>
          <p className="mt-0.5 text-[12px] text-navy-soft">
            <span className="font-semibold text-navy">Occupancy:</span> {operationalState.occupancy_percent}% |{" "}
            <span className="font-semibold text-navy">Waiting Queue:</span> {operationalState.patients_waiting} pts |{" "}
            <span className="font-semibold text-navy">Arrival Velocity:</span> {operationalState.arrival_rate} pts/hr |{" "}
            <span className="font-semibold text-navy">Staffing:</span> {operationalState.available_doctors} docs / {operationalState.available_nurses} nurses
          </p>
        </div>
      </div>

      <Link
        to="/dashboard"
        className="inline-flex items-center gap-1.5 rounded-xl border border-blue/30 bg-surface px-3 py-1.5 text-[12px] font-bold text-blue hover:bg-blue hover:text-white transition-colors shrink-0 shadow-sm"
      >
        <Sliders className="h-3.5 w-3.5" />
        <span>Modify in Overview</span>
        <ArrowRight className="h-3.5 w-3.5" />
      </Link>
    </div>
  );
}
