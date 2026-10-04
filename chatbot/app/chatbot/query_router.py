"""
==============================================================================
Query Router Module
==============================================================================
Categorizes incoming user queries into one of three distinct operational domains:
1. CATEGORY 1: REAL-TIME OPERATIONAL / PREDICTION QUESTIONS (ML Models)
2. CATEGORY 2: KNOWLEDGE-BASED QUESTIONS (ChromaDB RAG Retrieval)
3. CATEGORY 3: GENERAL CONVERSATIONAL QUESTIONS (Standard Bot Rules & Info)

Features explicit fallback handling when intent or confidence is ambiguous.
"""

import re
import logging
from enum import Enum
from typing import Dict, Any, Tuple, Optional

from app.schemas.prediction_schema import Intent

logger = logging.getLogger("chatbot.query_router")


class QueryCategory(str, Enum):
    """Supported query categories for the chatbot routing layer."""
    OPERATIONAL_PREDICTION = "OPERATIONAL_PREDICTION"
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    GENERAL_CONVERSATIONAL = "GENERAL_CONVERSATIONAL"


class QueryRouter:
    """
    Modular query classification router mapping user intents and query patterns
    to the appropriate execution pipeline.
    """

    OPERATIONAL_INTENTS = {
        Intent.PATIENT_VOLUME,
        Intent.WAITING_TIME,
        Intent.CROWDING,
        Intent.HIGH_DEMAND_PERIOD,
        Intent.FLOW_PATTERN,
        Intent.GENERAL_STATUS,
    }

    CONVERSATIONAL_INTENTS = {
        Intent.GREETING,
        Intent.HELP,
        Intent.MODEL_INFO,
        Intent.PROJECT_INFO,
        Intent.OUT_OF_SCOPE_MEDICAL,
    }

    KNOWLEDGE_PATTERNS = [
        r"\b(triage|esi|level\s*[1-5])\b",
        r"\b(policy|policies|protocol|protocols|guideline|guidelines|procedure)\b",
        r"\b(diversion|full\s*capacity|surge\s*management|hallway\s*beds?)\b",
        r"\b(causes\s*of|why\s*do|how\s*does\s*er|hospital\s*management)\b",
        r"\b(knowledge\s*base|documentation|system\s*architecture|manual)\b",
        r"what\s*is\s*(a\s*)?(triage|esi|crowding|diversion|capacity)",
        r"explain\s*(the\s*)?(triage|esi|crowding|policy|protocol|architecture)",
    ]

    def route_query(
        self,
        intent: Intent | str,
        text: str,
        confidence: float = 0.0
    ) -> QueryCategory:
        """
        Determines the query category for a user message.

        Args:
            intent (Intent | str): Resolved intent enum or string.
            text (str): Cleaned user message.
            confidence (float): Intent detection confidence score.

        Returns:
            QueryCategory: Category enum (OPERATIONAL_PREDICTION, KNOWLEDGE_BASE, GENERAL_CONVERSATIONAL)
        """
        if isinstance(intent, str):
            try:
                intent_enum = Intent(intent)
            except ValueError:
                intent_enum = Intent.UNKNOWN
        else:
            intent_enum = intent

        lower_text = text.lower().strip()

        # 1. CATEGORY 2: Knowledge Base & Advisory Queries (ESI triage, surge policies, staffing ratios, waiting time reduction strategies)
        is_knowledge_or_advisory = (
            getattr(Intent, "KNOWLEDGE_QUERY", None) == intent_enum
            or any(re.search(pattern, lower_text) for pattern in self.KNOWLEDGE_PATTERNS)
            or any(
                phrase in lower_text
                for phrase in [
                    "how can", "how to", "how do", "ways to", "how would", "tips to",
                    "strategy", "strategies", "tactic", "tactics", "best practice",
                    "reduce", "decrease", "cut", "shorten", "minimize", "prevent",
                    "causes of", "cause of", "why do", "why is", "why are", "explain",
                    "protocol", "policy", "guideline", "guidelines", "procedure",
                    "ratio", "staffing", "nurse", "doctor"
                ]
            )
        )
        if is_knowledge_or_advisory:
            logger.info(f"[QueryRouter] Query routed to CATEGORY 2 (KNOWLEDGE_BASE) - Knowledge/Advisory Query: '{text}'")
            return QueryCategory.KNOWLEDGE_BASE

        # 2. CATEGORY 1: Real-Time ML Predictions (High priority for operational point-estimate queries)
        if intent_enum in self.OPERATIONAL_INTENTS and confidence >= 0.50:
            logger.info(f"[QueryRouter] Query routed to CATEGORY 1 (OPERATIONAL_PREDICTION) - Intent: {intent_enum.value}")
            return QueryCategory.OPERATIONAL_PREDICTION

        # 3. CATEGORY 3: General Conversational & Informational Intents (Greetings, Help, Model Info, Project Info, Safety Refusals)
        if intent_enum in self.CONVERSATIONAL_INTENTS:
            logger.info(f"[QueryRouter] Query routed to CATEGORY 3 (GENERAL_CONVERSATIONAL) - Intent: {intent_enum.value}")
            return QueryCategory.GENERAL_CONVERSATIONAL

        # 4. Explicit Fallback: If intent is UNKNOWN or confidence is low, fall back safely to GENERAL_CONVERSATIONAL
        logger.info(f"[QueryRouter] Ambiguous or unknown query routed to fallback CATEGORY 3 (GENERAL_CONVERSATIONAL)")
        return QueryCategory.GENERAL_CONVERSATIONAL


# Global singleton instance
query_router = QueryRouter()
