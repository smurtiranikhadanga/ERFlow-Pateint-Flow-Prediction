import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  Bot,
  Clock,
  LayoutDashboard,
  PanelLeftClose,
  PanelLeftOpen,
  Server,
  TrendingUp,
  Waves,
  X,
  Zap,
} from "lucide-react";

export const NAV_ITEMS = [
  { to: "/dashboard", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/dashboard/forecast", label: "Patient Forecast", icon: TrendingUp },
  { to: "/dashboard/waiting-time", label: "Waiting Time", icon: Clock },
  { to: "/dashboard/crowding-risk", label: "Crowding Risk", icon: AlertTriangle },
  { to: "/dashboard/flow-patterns", label: "Flow Patterns", icon: Activity },
  { to: "/dashboard/surge-detection", label: "Surge Detection", icon: Waves },
  { to: "/dashboard/scenario-simulator", label: "Scenario Simulator", icon: Zap },
  { to: "/dashboard/monitoring", label: "Model Monitoring", icon: Server },
  { to: "/dashboard/ai-assistant", label: "AI Assistant", icon: Bot },
];

function Logo({ collapsed }) {
  return (
    <div className={`flex items-center gap-2.5 px-1 ${collapsed ? "justify-center" : ""}`}>
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue text-white">
        <Activity className="h-5 w-5" strokeWidth={2.25} aria-hidden="true" />
      </span>
      {!collapsed && (
        <div className="leading-tight">
          <p className="text-[15px] font-semibold tracking-tight text-white">ERFlow</p>
          <p className="text-[10.5px] font-medium uppercase tracking-[0.12em] text-white/50">
            Operations
          </p>
        </div>
      )}
    </div>
  );
}

function NavItems({ collapsed, onNavigate }) {
  return (
    <ul className="flex flex-col gap-1">
      {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
        <li key={to}>
          <NavLink
            to={to}
            end={end}
            onClick={onNavigate}
            title={collapsed ? label : undefined}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-lg px-3 py-2.5 text-[14.5px] font-medium transition-colors ${
                collapsed ? "justify-center" : ""
              } ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-white/60 hover:bg-white/5 hover:text-white"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={`flex h-5 w-5 shrink-0 items-center justify-center ${
                    isActive ? "text-blue" : ""
                  }`}
                >
                  <Icon className="h-[18px] w-[18px]" strokeWidth={2.25} aria-hidden="true" />
                </span>
                {!collapsed && <span className="truncate">{label}</span>}
                {isActive && !collapsed && (
                  <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-blue" />
                )}
              </>
            )}
          </NavLink>
        </li>
      ))}
    </ul>
  );
}

// Desktop / tablet sidebar — dynamic hover expansion, collapsed when not hovered.
export function DesktopSidebar({ collapsed: externalCollapsed, onToggleCollapsed }) {
  const [isHovered, setIsHovered] = useState(false);
  const [isPinned, setIsPinned] = useState(false);

  // Collapsed when mouse is not hovering (unless pinned)
  const isCollapsed = isPinned ? externalCollapsed : !isHovered;

  return (
    <aside
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`sticky top-0 hidden h-svh shrink-0 flex-col justify-between border-r border-white/10 bg-navy px-3 py-4 transition-all duration-300 ease-in-out z-30 lg:flex ${
        isCollapsed ? "w-[76px]" : "w-64 shadow-2xl"
      }`}
    >
      <div>
        <div className="mb-6">
          <Logo collapsed={isCollapsed} />
        </div>
        <NavItems collapsed={isCollapsed} />
      </div>

      <button
        type="button"
        onClick={() => {
          setIsPinned((prev) => !prev);
          if (onToggleCollapsed) onToggleCollapsed();
        }}
        aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        className={`flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-[13.5px] font-medium text-white/60 transition-colors hover:bg-white/5 hover:text-white ${
          isCollapsed ? "justify-center" : ""
        }`}
      >
        {isCollapsed ? (
          <PanelLeftOpen className="h-[18px] w-[18px]" strokeWidth={2.25} aria-hidden="true" />
        ) : (
          <>
            <PanelLeftClose className="h-[18px] w-[18px]" strokeWidth={2.25} aria-hidden="true" />
            <span>{isPinned ? "Unpin" : "Pin"}</span>
          </>
        )}
      </button>
    </aside>
  );
}

// Mobile sidebar — slides in as an overlay drawer.
export function MobileSidebar({ open, onClose }) {
  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-navy/40 transition-opacity lg:hidden ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        style={{ backdropFilter: open ? "blur(1px)" : undefined }}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex h-svh w-72 max-w-[82vw] flex-col justify-between bg-navy px-3 py-4 shadow-lift transition-transform duration-200 lg:hidden ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
        aria-hidden={!open}
      >
        <div>
          <div className="mb-6 flex items-center justify-between px-1">
            <Logo collapsed={false} />
            <button
              type="button"
              onClick={onClose}
              aria-label="Close menu"
              className="flex h-8 w-8 items-center justify-center rounded-lg text-white/70 hover:bg-white/10 hover:text-white"
            >
              <X className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
          <NavItems collapsed={false} onNavigate={onClose} />
        </div>
      </aside>
    </>
  );
}
