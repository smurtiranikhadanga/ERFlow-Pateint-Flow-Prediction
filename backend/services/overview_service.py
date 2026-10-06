import logging

from .supervised_service import supervised_service
from .unsupervised_service import unsupervised_service
from .deep_learning_service import deep_learning_service
from ..schemas.hospital_state import HospitalState
from ..schemas.overview import (
    DashboardOverviewResponse,
    AssistantQueryRequest,
    AssistantQueryResponse,
    AssistantInsightItem,
)

logger = logging.getLogger("erflow.overview_service")


class OverviewService:
    """Combines outputs from all three ML pillars for the Overview dashboard and AI Assistant."""

    def get_overview(self, state: HospitalState) -> DashboardOverviewResponse:
        if isinstance(state, dict):
            state = HospitalState(**state)
        forecast = deep_learning_service.forecast_arrivals(state)
        waiting = supervised_service.predict_waiting_time(state)
        crowding = supervised_service.predict_crowding_risk(state)
        flow = unsupervised_service.predict_flow_pattern(state)
        surge = unsupervised_service.detect_surge(state)

        summary_text = (
            f"Patient demand is forecasted at {forecast.horizons['3h']} arrivals over the next 3 hours. "
            f"Expected waiting time is currently {waiting.waiting_time_minutes:.0f} minutes with a {crowding.crowding_level} "
            f"crowding risk (score: {crowding.crowding_score}/100). Current flow pattern reflects {flow.pattern_name} "
            f"({flow.confidence:.0f}% confidence) with {surge.status.lower()}."
        )

        return DashboardOverviewResponse(
            forecast=forecast,
            waiting_time=waiting,
            crowding_risk=crowding,
            flow_pattern=flow,
            surge_detection=surge,
            ai_summary_text=summary_text
        )

    def answer_assistant_query(self, query_req: AssistantQueryRequest) -> AssistantQueryResponse:
        state_dict = query_req.hospital_state or {}
        state = HospitalState(**state_dict) if state_dict else HospitalState()

        overview = self.get_overview(state)
        q = query_req.question.lower()

        # Dynamic query routing
        if "busiest" in q or "peak" in q or "when" in q:
            text = (
                f"Patient arrivals are expected to peak around {overview.forecast.predicted_peak_time} "
                f"with an arrival velocity of {overview.forecast.predicted_peak_rate} patients/hour. "
                f"The next 3-hour projection indicates {overview.forecast.horizons['3h']} cumulative arrivals, "
                f"bringing crowding risk to {overview.crowding_risk.crowding_level}."
            )
            insights = [
                AssistantInsightItem(label="Expected Arrivals", value=str(overview.forecast.horizons["3h"]), icon="Users", tone="blue"),
                AssistantInsightItem(label="Peak Time", value=overview.forecast.predicted_peak_time, icon="Clock", tone="teal"),
                AssistantInsightItem(label="Crowding Risk", value=overview.crowding_risk.crowding_level, icon="AlertTriangle", tone="red" if overview.crowding_risk.crowding_level in ["HIGH", "CRITICAL"] else "amber"),
                AssistantInsightItem(label="Expected Wait", value=f"{overview.waiting_time.waiting_time_minutes:.0f} min", icon="Timer", tone="amber"),
            ]
        elif "wait" in q or "time" in q:
            text = (
                f"Average waiting time is estimated at {overview.waiting_time.waiting_time_minutes:.0f} minutes by the XGBoost Regressor. "
                f"Wait time is {overview.waiting_time.trend.lower()} and projected to reach {overview.waiting_time.predicted_1h:.0f} minutes "
                f"over the next hour with bed occupancy at {state.occupancy_percent:.0f}%."
            )
            insights = [
                AssistantInsightItem(label="Expected Wait", value=f"{overview.waiting_time.waiting_time_minutes:.0f} min", icon="Timer", tone="amber"),
                AssistantInsightItem(label="Wait Trend", value=overview.waiting_time.trend, icon="TrendingUp", tone="red" if overview.waiting_time.trend == "Increasing" else "teal"),
                AssistantInsightItem(label="Projected 1h", value=f"{overview.waiting_time.predicted_1h:.0f} min", icon="Clock", tone="blue"),
                AssistantInsightItem(label="Beds Occupied", value=f"{state.occupancy_percent:.0f}%", icon="Activity", tone="teal"),
            ]
        elif "surge" in q or "spike" in q:
            text = (
                f"{'Abnormal patient surge detected!' if overview.surge_detection.is_surge else 'No anomalous surge detected.'} "
                f"Current arrival rate is {overview.surge_detection.current_arrival_rate:.0f} patients/hour versus the expected baseline "
                f"of {overview.surge_detection.normal_arrival_rate} ({overview.surge_detection.deviation_percent})."
            )
            insights = [
                AssistantInsightItem(label="Surge Status", value=overview.surge_detection.severity, icon="AlertTriangle", tone="red" if overview.surge_detection.is_surge else "green"),
                AssistantInsightItem(label="Arrival Rate", value=f"{overview.surge_detection.current_arrival_rate:.0f}/hr", icon="TrendingUp", tone="amber"),
                AssistantInsightItem(label="Deviation", value=overview.surge_detection.deviation_percent, icon="Activity", tone="blue"),
                AssistantInsightItem(label="Baseline", value=f"{overview.surge_detection.normal_arrival_rate}/hr", icon="Clock", tone="teal"),
            ]
        elif "flow" in q or "pattern" in q:
            text = (
                f"The ER is exhibiting a '{overview.flow_pattern.pattern_name}' pattern with {overview.flow_pattern.confidence:.0f}% "
                f"confidence. {overview.flow_pattern.description}"
            )
            insights = [
                AssistantInsightItem(label="Flow Pattern", value=overview.flow_pattern.pattern_name, icon="Activity", tone="blue"),
                AssistantInsightItem(label="Confidence", value=f"{overview.flow_pattern.confidence:.0f}%", icon="Cpu", tone="teal"),
                AssistantInsightItem(label="Peak Time", value=overview.forecast.predicted_peak_time, icon="Clock", tone="amber"),
                AssistantInsightItem(label="Crowding Risk", value=overview.crowding_risk.crowding_level, icon="AlertTriangle", tone="red" if overview.crowding_risk.crowding_level in ["HIGH", "CRITICAL"] else "amber"),
            ]
        elif "crowding" in q or "cause" in q or "risk" in q:
            text = (
                f"Crowding risk is {overview.crowding_risk.crowding_level} with an index of {overview.crowding_risk.crowding_score}/100. "
                f"Key drivers are current occupancy ({state.occupancy_percent:.0f}%), {state.patients_waiting:.0f} waiting patients, "
                f"and forecasted arrivals of {overview.forecast.horizons['3h']} patients over the next 3 hours."
            )
            insights = [
                AssistantInsightItem(label="Crowding Score", value=f"{overview.crowding_risk.crowding_score}/100", icon="AlertTriangle", tone="red"),
                AssistantInsightItem(label="Bed Occupancy", value=f"{state.occupancy_percent:.0f}%", icon="Activity", tone="amber"),
                AssistantInsightItem(label="Patients Waiting", value=str(int(state.patients_waiting)), icon="Users", tone="blue"),
                AssistantInsightItem(label="Available Beds", value=str(int(state.available_beds)), icon="Activity", tone="teal"),
            ]
        else:
            text = (
                f"Operational Status: The ER is experiencing {overview.flow_pattern.pattern_name.lower()} conditions. "
                f"Predicted arrivals over next 3h: {overview.forecast.horizons['3h']}. Average wait time: {overview.waiting_time.waiting_time_minutes:.0f} min. "
                f"Overall crowding level: {overview.crowding_risk.crowding_level}."
            )
            insights = [
                AssistantInsightItem(label="Wait Time", value=f"{overview.waiting_time.waiting_time_minutes:.0f} min", icon="Timer", tone="amber"),
                AssistantInsightItem(label="Crowding", value=overview.crowding_risk.crowding_level, icon="AlertTriangle", tone="red" if overview.crowding_risk.crowding_level in ["HIGH", "CRITICAL"] else "teal"),
                AssistantInsightItem(label="3h Forecast", value=str(overview.forecast.horizons["3h"]), icon="Users", tone="blue"),
                AssistantInsightItem(label="Pattern", value=overview.flow_pattern.pattern_name, icon="Activity", tone="teal"),
            ]

        return AssistantQueryResponse(text=text, insights=insights)


overview_service = OverviewService()
