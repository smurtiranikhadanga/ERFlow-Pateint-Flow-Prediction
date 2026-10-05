import { SlidersHorizontal, Play, RotateCcw } from "lucide-react";
import StepperControl from "./StepperControl";

export const DEFAULT_HOSPITAL_STATE = {
  hour_of_day: 18,
  day_of_week: 4, // 0=Mon, 4=Fri
  month: 7, // July
  arrival_rate: 28,
  available_beds: 8,
  available_doctors: 5,
  available_nurses: 9,
  patients_waiting: 24,
  severity_level: 3,
  occupancy_percent: 78,
};

const DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
];
const SEVERITY_NAMES = [
  { val: 1, label: "Level 1 — Resuscitation (Most Severe)" },
  { val: 2, label: "Level 2 — Emergent" },
  { val: 3, label: "Level 3 — Urgent" },
  { val: 4, label: "Level 4 — Less Urgent" },
  { val: 5, label: "Level 5 — Non-Urgent (Least Severe)" },
];

export default function HospitalStateForm({ onSubmit, loading, initialValues }) {
  const [form, setForm] = useState({ ...DEFAULT_HOSPITAL_STATE, ...initialValues });
  const [errors, setErrors] = useState({});
  const [collapsed, setCollapsed] = useState(false);

  function handleChange(field, val) {
    setForm((prev) => {
      const updated = { ...prev, [field]: val };
      if (field === "day_of_week") {
        updated.is_weekend = val >= 5 ? 1 : 0;
      }
      return updated;
    });
    // Clear field error on change
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: null }));
    }
  }

  function validate() {
    const errs = {};
    if (form.arrival_rate < 0 || isNaN(form.arrival_rate)) errs.arrival_rate = "Must be ≥ 0";
    if (form.available_beds < 0 || isNaN(form.available_beds)) errs.available_beds = "Must be ≥ 0";
    if (form.available_doctors < 0 || isNaN(form.available_doctors)) errs.available_doctors = "Must be ≥ 0";
    if (form.available_nurses < 0 || isNaN(form.available_nurses)) errs.available_nurses = "Must be ≥ 0";
    if (form.patients_waiting < 0 || isNaN(form.patients_waiting)) errs.patients_waiting = "Must be ≥ 0";
    if (form.occupancy_percent < 0 || form.occupancy_percent > 100 || isNaN(form.occupancy_percent)) {
      errs.occupancy_percent = "Must be between 0 and 100%";
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!validate()) return;
    onSubmit(form);
  }

  function handleReset() {
    setForm(DEFAULT_HOSPITAL_STATE);
    setErrors({});
    onSubmit(DEFAULT_HOSPITAL_STATE);
  }

  return (
    <div className="rounded-2xl border border-border bg-surface shadow-soft transition-all">
      <div className="flex items-center justify-between border-b border-border px-5 py-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-tint text-blue">
            <SlidersHorizontal className="h-4 w-4" strokeWidth={2.25} />
          </span>
          <div>
            <h3 className="text-[15px] font-semibold text-navy">Live ER Operational Inputs</h3>
            <p className="text-[12px] text-navy-soft">
              Tweak hospital variables to test real-time ML model predictions
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          className="text-[12.5px] font-semibold text-blue hover:text-blue-dark"
        >
          {collapsed ? "Expand Controls" : "Collapse"}
        </button>
      </div>

      {!collapsed && (
        <form onSubmit={handleSubmit} className="p-5 sm:p-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {/* Hour of Day */}
            <StepperControl
              label="Hour of Day"
              value={form.hour_of_day}
              onChange={(val) => handleChange("hour_of_day", val)}
              min={0}
              max={23}
              step={1}
              unit=":00"
            />

            {/* Day of Week */}
            <div>
              <label className="block text-[12px] font-semibold uppercase tracking-wider text-navy-soft">
                Day of Week
              </label>
              <select
                value={form.day_of_week}
                onChange={(e) => handleChange("day_of_week", parseInt(e.target.value, 10))}
                className="mt-1.5 w-full rounded-lg border border-border bg-bg px-3 py-2 text-[13px] font-medium text-navy focus:border-blue focus:outline-none"
              >
                {DAY_NAMES.map((name, idx) => (
                  <option key={name} value={idx}>
                    {name} {idx >= 5 ? "(Weekend)" : "(Weekday)"}
                  </option>
                ))}
              </select>
            </div>

            {/* Month */}
            <div>
              <label className="block text-[12px] font-semibold uppercase tracking-wider text-navy-soft">
                Month
              </label>
              <select
                value={form.month}
                onChange={(e) => handleChange("month", parseInt(e.target.value, 10))}
                className="mt-1.5 w-full rounded-lg border border-border bg-bg px-3 py-2 text-[13px] font-medium text-navy focus:border-blue focus:outline-none"
              >
                {MONTH_NAMES.map((name, idx) => (
                  <option key={name} value={idx + 1}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            {/* Arrival Rate */}
            <StepperControl
              label="Arrival Rate"
              value={form.arrival_rate}
              onChange={(val) => handleChange("arrival_rate", val)}
              min={0}
              max={100}
              step={1}
              unit="pts/hr"
              error={errors.arrival_rate}
            />

            {/* Available Beds */}
            <StepperControl
              label="Available Staffed Beds"
              value={form.available_beds}
              onChange={(val) => handleChange("available_beds", val)}
              min={0}
              max={50}
              step={1}
              error={errors.available_beds}
            />

            {/* Available Doctors */}
            <StepperControl
              label="Available Doctors"
              value={form.available_doctors}
              onChange={(val) => handleChange("available_doctors", val)}
              min={0}
              max={20}
              step={1}
              error={errors.available_doctors}
            />

            {/* Available Nurses */}
            <StepperControl
              label="Available Nurses"
              value={form.available_nurses}
              onChange={(val) => handleChange("available_nurses", val)}
              min={0}
              max={40}
              step={1}
              error={errors.available_nurses}
            />

            {/* Patients Waiting */}
            <StepperControl
              label="Patients Waiting in Triage"
              value={form.patients_waiting}
              onChange={(val) => handleChange("patients_waiting", val)}
              min={0}
              max={100}
              step={1}
              error={errors.patients_waiting}
            />

            {/* Severity Level */}
            <div>
              <label className="block text-[12px] font-semibold uppercase tracking-wider text-navy-soft">
                Average Acuity / Severity
              </label>
              <select
                value={form.severity_level}
                onChange={(e) => handleChange("severity_level", parseInt(e.target.value, 10))}
                className="mt-1.5 w-full rounded-lg border border-border bg-bg px-3 py-2 text-[13px] font-medium text-navy focus:border-blue focus:outline-none"
              >
                {SEVERITY_NAMES.map((s) => (
                  <option key={s.val} value={s.val}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Occupancy Percent */}
            <StepperControl
              label="Bed Occupancy Rate"
              value={form.occupancy_percent}
              onChange={(val) => handleChange("occupancy_percent", val)}
              min={0}
              max={100}
              step={1}
              unit="%"
              error={errors.occupancy_percent}
            />
          </div>

          <div className="mt-5 flex items-center justify-end gap-3 border-t border-border pt-4">
            <button
              type="button"
              onClick={handleReset}
              disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-xl border border-border px-4 py-2 text-[13px] font-semibold text-navy-soft transition-colors hover:bg-bg hover:text-navy disabled:opacity-50"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset Defaults
            </button>
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-xl bg-blue px-5 py-2 text-[13px] font-semibold text-white shadow-soft transition-colors hover:bg-blue-dark disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              {loading ? "Calculating Prediction..." : "Run Model Inference"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
