import re
from typing import Any, Dict, List, Tuple
from app.schemas.prediction_schema import Intent


class IntentDetector:
    """
    Rule-based intent detector for the Emergency Room Patient Flow Prediction system.
    Supports all 12 operational and conversational intents with dynamic confidence
    scoring and priority ordering without requiring external LLM dependencies.
    """

    # 1. OUT_OF_SCOPE_MEDICAL (Intercepts explicit personal clinical self-diagnosis or acute medical treatment requests)
    OUT_OF_SCOPE_PATTERNS = [
        (r"\b(i\s*have|i\s*am\s*having|my|feel).*?(chest\s*pain|shortness\s*of\s*breath|heart\s*attack|severe\s*headache|fever|cough|dizziness)\b", 0.95),
        (r"\b(diagnose\s*me|what\s*(disease|illness)\s*do\s*i\s*have|do\s*i\s*have)\b", 0.95),
        (r"\b(what\s*(medicine|drug|pill|dosage)\s*should\s*i\s*take|prescribe\s*me)\b", 0.95),
        (r"\b(how\s*to\s*treat|home\s*remedy\s*for|medical\s*advice)\b", 0.95),
        (r"\b(symptoms?\s*of\s*(a\s*)?(heart\s*attack|stroke)|is\s*my\s*.*?\s*life\s*threatening)\b", 0.95),
    ]

    # 2. GREETING PATTERNS
    GREETING_PATTERNS = [
        (r"^(hi|hello|hey|greetings)\b", 0.95),
        (r"^good\s*(morning|afternoon|evening|day)\b", 0.95),
    ]

    # 3. HELP PATTERNS
    HELP_PATTERNS = [
        (r"^help\b", 0.95),
        (r"what\s*can\s*you\s*do", 0.95),
        (r"how\s*to\s*use", 0.92),
        (r"\b(commands|instructions|options|menu)\b", 0.90),
    ]

    # 4. MODEL INFO PATTERNS
    MODEL_INFO_PATTERNS = [
        (r"what\s*model\s*(are\s*you\s*using|do\s*you\s*use|is\s*this|do\s*we\s*have)", 0.95),
        (r"how\s*does\s*(the\s*)?(lstm|xgboost|dbscan|kmeans)\s*model\s*work", 0.95),
        (r"are\s*(the\s*)?models\s*working", 0.95),
        (r"how\s*does\s*(the\s*)?(prediction\s*)?model\s*work", 0.94),
        (r"what\s*(ml\s*)?model\s*(is\s*used|do\s*you\s*use)", 0.94),
        (r"explain\s*(the\s*)?(ml\s*)?model", 0.92),
        (r"model\s*(architecture|accuracy|algorithm|details|info)", 0.90),
        (r"\b(lstm|xgboost|dbscan|kmeans)\s*model\b", 0.92),
        (r"how\s*(do|are)\s*(you|predictions)\s*(predict|made|forecast)", 0.90),
    ]

    # 5. PROJECT INFO PATTERNS
    PROJECT_INFO_PATTERNS = [
        (r"what\s*does\s*this\s*project\s*do", 0.94),
        (r"tell\s*me\s*about\s*(this\s*)?project", 0.94),
        (r"about\s*(this\s*)?(project|system|application|chatbot)", 0.92),
        (r"project\s*(info|details|purpose|overview|scope)", 0.90),
        (r"who\s*built\s*this", 0.90),
    ]

    # 5b. KNOWLEDGE QUERY PATTERNS
    KNOWLEDGE_PATTERNS = [
        (r"\b(what\s*is|explain|tell\s*me\s*about)\s*(er\s*)?triage\b", 0.95),
        (r"\b(esi\s*levels?|emergency\s*severity\s*index|acuity\s*level)\b", 0.95),
        (r"\b(diversion\s*policy|full\s*capacity\s*protocol|hallway\s*beds?)\b", 0.95),
        (r"\b(causes\s*of|reasons\s*for)\s*(er\s*)?(overcrowding|crowding|delays)", 0.95),
        (r"\b(knowledge\s*base|hospital\s*guidelines|triage\s*protocols?)\b", 0.95),
    ]

    # 6. WAITING TIME PATTERNS
    WAITING_TIME_PATTERNS = [
        (r"why\s*is\s*(the\s*)?wait(ing)?([\s\-]*time)?\s*(so\s*)?(high|long|elevated)?", 0.95),
        (r"what\s*factors\s*are\s*causing\s*(the\s*)?(long\s*wait|high\s*waiting\s*time)", 0.95),
        (r"why\s*(are|is)\s*(the\s*)?queue\s*(so\s*)?long", 0.95),
        (r"wait(ing)?[\s\-]*(time|duration|estimate|minutes|risk|situation|period|level)", 0.95),
        (r"how\s*long\s*(will|do|are|is)\s*(patients|people|i)?\s*(have\s*to\s*)?(wait|waiting)", 0.95),
        (r"what\s*(will|is)\s*(the\s*)?(current|expected|estimated\s*)?wait(ing)?[\s\-]*time", 0.95),
        (r"how\s*long\s*is\s*(the\s*)?(current\s*)?(wait|queue)", 0.94),
        (r"(are|is)\s*(patients\s*)?waiting\s*(longer|more|high|increasing)", 0.94),
        (r"(is\s*the\s*)?queue\s*(getting\s*)?(longer|growing|increasing|time)", 0.93),
        (r"how\s*bad\s*is\s*(the\s*)?wait", 0.93),
        (r"should\s*i\s*expect\s*a\s*(long\s*)?wait", 0.92),
        (r"(er|ed)?\s*wait[\s\-]*time", 0.92),
        (r"triage\s*(wait|time)", 0.90),
        (r"queue\s*(duration|length|status)", 0.90),
        (r"\bdelay(s)?\b", 0.80),
    ]

    # 7. PATIENT VOLUME PATTERNS
    PATIENT_VOLUME_PATTERNS = [
        (r"what\s*is\s*(the\s*)?(current\s*)?patient\s*arrival\s*rate", 0.95),
        (r"arrival\s*rate", 0.95),
        (r"is\s*demand\s*(increasing|growing|rising)", 0.95),
        (r"what\s*will\s*patient\s*arrivals\s*look\s*like", 0.95),
        (r"how\s*many\s*patients(\s*are)?\s*(expected|predicted)", 0.95),
        (r"expected\s*(patient|arrival|admission)\s*(count|volume|rate)", 0.94),
        (r"patient\s*volume(\s*forecast)?", 0.92),
        (r"how\s*many\s*(arrivals|admissions|patients)", 0.92),
        (r"patient\s*arrivals?", 0.90),
        (r"admissions\s*forecast", 0.90),
        (r"volume\s*forecast", 0.88),
    ]

    # 8. FLOW PATTERN PATTERNS
    FLOW_PATTERN_PATTERNS = [
        (r"what\s*flow\s*pattern\s*are\s*we\s*seeing", 0.95),
        (r"what\s*patterns?\s*(are\s*)?(you\s*seeing|present|there|in)", 0.95),
        (r"what\s*patterns?\s*(do\s*you\s*see\s*in\s*)?patient\s*flow", 0.95),
        (r"are\s*there\s*(unusual|any)\s*patient[- ]flow\s*patterns?", 0.94),
        (r"(operational|current)\s*pattern(s)?", 0.94),
        (r"patient[- ]flow\s*pattern(s)?", 0.93),
        (r"flow\s*pattern(s)?", 0.90),
        (r"patient[- ]flow\s*regime", 0.90),
        (r"\bk-means\b", 0.90),
        (r"\bcluster(ing)?\b", 0.88),
        (r"demand\s*regime", 0.88),
    ]

    # 9. HIGH DEMAND PERIOD PATTERNS
    HIGH_DEMAND_PATTERNS = [
        (r"is\s*there\s*a\s*surge", 0.95),
        (r"when\s*(will|is)\s*(the\s*)?(er|ed|emergency\s*room|emergency\s*department)\s*(be\s*)?(busiest|peak)", 0.95),
        (r"are\s*we\s*experiencing\s*a\s*high[\s\-]demand", 0.95),
        (r"are\s*there\s*(any\s*)?surge\s*(periods?|hours?|times?)", 0.94),
        (r"busiest\s*(time|period|hours?|day)", 0.93),
        (r"high\s*demand\s*(period|hours?|time|surge)?", 0.92),
        (r"peak\s*(period|hours?|time|surge|demand)", 0.90),
        (r"\bsurge\s*(period|hours?|time)\b", 0.90),
        (r"rush\s*hours?", 0.85),
    ]

    # 10. CROWDING PATTERNS
    CROWDING_PATTERNS = [
        (r"why\s*is\s*(the\s*)?crowding\s*(high|critical|severe|elevated)?", 0.95),
        (r"what\s*factors\s*are\s*causing\s*(the\s*)?crowding", 0.95),
        (r"why\s*is\s*(the\s*)?(er|ed|emergency\s*room)\s*(so\s*)?busy", 0.95),
        (r"will\s*(the\s*)?(er|ed|emergency\s*room|emergency\s*department)?\s*(crowding\s*)?(increase|grow|rise|be\s*crowded)", 0.95),
        (r"is\s*(the\s*)?(er|ed|emergency\s*room|emergency\s*department)\s*crowded", 0.95),
        (r"current\s*crowding\s*(level|status|state|risk)?", 0.95),
        (r"how\s*busy\s*is\s*(the\s*)?(emergency\s*room|emergency\s*department)", 0.94),
        (r"crowd(ing|ed)?", 0.88),
        (r"occupancy(\s*rate)?", 0.88),
        (r"full\s*capacity", 0.88),
        (r"bed\s*availability", 0.85),
        (r"er\s*capacity", 0.85),
        (r"congestion", 0.82),
    ]

    # 11. GENERAL STATUS PATTERNS
    GENERAL_STATUS_PATTERNS = [
        (r"what\s*(should|needs|do)\s*(the\s*)?(er|ed|emergency\s*room)?\s*(staff\s*)?(pay\s*)?attention\s*(to)?", 0.95),
        (r"what\s*needs\s*attention", 0.95),
        (r"attention\s*(required|needed)", 0.95),
        (r"how\s*is\s*(the\s*)?(er|ed|emergency\s*room)\s*(doing|operating)\s*(right\s*now)?", 0.95),
        (r"give\s*me\s*a\s*summary\s*of\s*(the\s*)?(current\s*)?(er|ed)", 0.95),
        (r"what\s*is\s*causing\s*(the\s*)?current\s*pressure", 0.95),
        (r"how\s*busy\s*is\s*(the\s*)?(er|ed)", 0.93),
        (r"how\s*is\s*(the\s*)?(er|ed|emergency\s*room|emergency\s*department)\s*(expected\s*to\s*be|today|right\s*now|tonight|going)", 0.92),
        (r"(er|ed|emergency\s*room)\s*(general\s*)?status", 0.90),
        (r"general\s*status", 0.88),
        (r"overall\s*status", 0.88),
        (r"patient\s*flow\s*(overview|summary|status)", 0.88),
        (r"daily\s*(overview|summary)", 0.85),
        (r"flow\s*metrics", 0.85),
    ]

    def detect_intent(self, text: str) -> Dict[str, Any]:
        """
        Analyzes input text using structured pattern priority matching and dynamic confidence scoring.

        Returns:
            dict: {
                "intent": str (e.g. "PATIENT_VOLUME", "WAITING_TIME", ...),
                "confidence": float (0.0 to 1.0)
            }
        """
        if not text or not text.strip():
            return {
                "intent": Intent.UNKNOWN.value,
                "confidence": 0.0,
            }

        cleaned = text.lower().strip()
        words = cleaned.split()
        word_count = len(words)

        # Evaluator priority table
        evaluators: List[Tuple[Intent, List[Tuple[str, float]]]] = [
            (Intent.OUT_OF_SCOPE_MEDICAL, self.OUT_OF_SCOPE_PATTERNS),
            (Intent.GREETING, self.GREETING_PATTERNS),
            (Intent.HELP, self.HELP_PATTERNS),
            (Intent.MODEL_INFO, self.MODEL_INFO_PATTERNS),
            (Intent.PROJECT_INFO, self.PROJECT_INFO_PATTERNS),
            (Intent.KNOWLEDGE_QUERY, self.KNOWLEDGE_PATTERNS),
            (Intent.WAITING_TIME, self.WAITING_TIME_PATTERNS),
            (Intent.PATIENT_VOLUME, self.PATIENT_VOLUME_PATTERNS),
            (Intent.FLOW_PATTERN, self.FLOW_PATTERN_PATTERNS),
            (Intent.HIGH_DEMAND_PERIOD, self.HIGH_DEMAND_PATTERNS),
            (Intent.CROWDING, self.CROWDING_PATTERNS),
            (Intent.GENERAL_STATUS, self.GENERAL_STATUS_PATTERNS),
        ]

        best_intent = Intent.UNKNOWN
        best_confidence = 0.0

        for intent_enum, patterns in evaluators:
            for pattern, base_conf in patterns:
                match = re.search(pattern, cleaned)
                if match:
                    conf = base_conf

                    # Ambiguity penalty for single vague words
                    if word_count == 1 and not (intent_enum in [Intent.GREETING, Intent.HELP]):
                        matched_str = match.group(0)
                        if len(matched_str) < 8 or matched_str in ["time", "busy", "surge", "peak", "queue", "delay", "volume", "flow", "er", "ed"]:
                            conf = 0.45  # Penalty forces UNKNOWN classification for vague single words

                    if conf > best_confidence:
                        best_confidence = conf
                        best_intent = intent_enum
                    break

        if best_confidence < 0.50:
            return {
                "intent": Intent.UNKNOWN.value,
                "confidence": round(float(best_confidence) if best_confidence > 0 else 0.20, 2),
            }

        return {
            "intent": best_intent.value,
            "confidence": round(float(best_confidence), 2),
        }


# Global singleton instance
intent_detector = IntentDetector()
