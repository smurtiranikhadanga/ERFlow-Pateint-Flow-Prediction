import { Routes, Route, useLocation } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import DashboardLayout from "./dashboard/DashboardLayout";
import Overview from "./dashboard/pages/Overview";
import PatientForecast from "./dashboard/pages/PatientForecast";
import WaitingTime from "./dashboard/pages/WaitingTime";
import CrowdingRisk from "./dashboard/pages/CrowdingRisk";
import FlowPatterns from "./dashboard/pages/FlowPatterns";
import SurgeDetection from "./dashboard/pages/SurgeDetection";
import ScenarioSimulator from "./dashboard/pages/ScenarioSimulator";
import ModelMonitoring from "./dashboard/pages/ModelMonitoring";
import AIAssistant from "./dashboard/pages/AIAssistant";
import NotFound from "./pages/NotFound";
import ErrorBoundary from "./dashboard/components/ErrorBoundary";
import { ThemeProvider } from "./context/ThemeContext";
import { ModeProvider } from "./context/ModeContext";
import { ERProvider } from "./context/ERContext";

function AppContent() {
  const location = useLocation();
  const resetKey = location.pathname;

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />

      <Route path="/dashboard" element={<DashboardLayout />}>
        <Route index element={<ErrorBoundary resetKey={resetKey}><Overview /></ErrorBoundary>} />
        <Route path="forecast" element={<ErrorBoundary resetKey={resetKey}><PatientForecast /></ErrorBoundary>} />
        <Route path="waiting-time" element={<ErrorBoundary resetKey={resetKey}><WaitingTime /></ErrorBoundary>} />
        <Route path="crowding-risk" element={<ErrorBoundary resetKey={resetKey}><CrowdingRisk /></ErrorBoundary>} />
        <Route path="flow-patterns" element={<ErrorBoundary resetKey={resetKey}><FlowPatterns /></ErrorBoundary>} />
        <Route path="surge-detection" element={<ErrorBoundary resetKey={resetKey}><SurgeDetection /></ErrorBoundary>} />
        <Route path="scenario-simulator" element={<ErrorBoundary resetKey={resetKey}><ScenarioSimulator /></ErrorBoundary>} />
        <Route path="monitoring" element={<ErrorBoundary resetKey={resetKey}><ModelMonitoring /></ErrorBoundary>} />
        <Route path="ai-assistant" element={<ErrorBoundary resetKey={resetKey}><AIAssistant /></ErrorBoundary>} />
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

function App() {
  return (
    <ThemeProvider>
      <ModeProvider>
        <ERProvider>
          <AppContent />
        </ERProvider>
      </ModeProvider>
    </ThemeProvider>
  );
}

export default App;
