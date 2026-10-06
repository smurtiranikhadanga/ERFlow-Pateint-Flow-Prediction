from dataclasses import dataclass
import re
from typing import List, Optional
from app.schemas.prediction_schema import Intent


@dataclass
class SafetyCheckResult:
    """Represents the output of the safety and scope verification."""
    is_safe: bool
    reason: Optional[str] = None
    refusal_message: Optional[str] = None


class SafetyGuard:
    """
    Safety and Scope Enforcement Layer for the ER Patient Flow Chatbot.

    Purpose & Boundary:
    -------------------
    The chatbot is designed exclusively for Emergency Room operational predictions:
      - ER patient volume & arrival forecasting
      - Patient waiting times by triage acuity
      - Department crowding & bed occupancy
      - High-demand surge windows & operational peak periods
      - System & project information

    The chatbot strictly refuses:
      - Patient diagnosis & symptom assessments
      - Medication, prescription, or dosage advice
      - Medical treatment plans & clinical interventions
      - Direct medical emergency triage instructions
    """

    SAFETY_REFUSAL_MESSAGE = (
        "⚠️ **Medical Scope Notice**: I am an AI assistant designed strictly for **Emergency Room Operational "
        "and Patient Flow Predictions** (e.g., patient volume, estimated waiting times, and department crowding). "
        "I cannot provide medical diagnosis, treatment advice, medication recommendations, or individual clinical evaluations.\n\n"
        "If you or someone nearby is experiencing a medical emergency, please contact your local emergency services "
        "(e.g., 911 / 112) or consult a qualified healthcare professional immediately."
    )

    # 1. Symptom Checking & Diagnosis Inquiries (Targeted to First-Person Acute Requests)
    DIAGNOSIS_PATTERNS = [
        r"\b(diagnos(e|is)\s*(me|my)?|do\s*i\s*have|what\s*(disease|illness|condition|infection)\s*(do\s*i\s*have|is\s*this)?)\b",
        r"\b(i\s*(have|am\s*having|feel).*?(chest\s*pain|shortness\s*of\s*breath|severe\s*headache|dizziness|fever|cough|nausea|vomiting|bleeding))\b",
        r"\bmy\s*(chest|head|stomach|throat|arm|leg|abdomen|back|heart|eye)\s*(hurts|is\s*hurting|aches|pains)\b",
        r"\b(am\s*i\s*dying|is\s*my\s*condition\s*(fatal|cancer))\b",
        r"\b(symptoms?\s*of|signs?\s*of)\b",
        r"\b(life\s*threatening|fatal|fatalit(y|ies))\b",
        r"\b(abdominal|chest|head|stomach)\s*pain\b",
        r"\b(heart\s*attack|stroke|appendicitis)\b",
    ]

    # 2. Medication, Prescription, and Dosage Inquiries
    MEDICATION_PATTERNS = [
        r"\b(what\s*(medicine|medication|drug|pill|antibiotic|syrup)\s*(should|can|do)\s*i\s*take)\b",
        r"\b(prescribe|recommend\s*(a\s*)?(medicine|medication|drug|pill|treatment))\b",
        r"\b(how\s*much|what\s*dosage|how\s*many\s*mg|dose\s*of)\s*(paracetamol|ibuprofen|tylenol|aspirin|amoxicillin|insulin|antibiotics?|advil)\b",
        r"\bside\s*effects\s*of\s*(medications?|drugs?|antibiotics?)\b",
        r"\bcan\s*i\s*take\s*.*\s*with\s*.*\b",
    ]

    # 3. Treatment, Clinical Advice, and Home Remedies
    TREATMENT_PATTERNS = [
        r"\bhow\s*(to|do\s*i|can\s*i)\s*(treat|cure|heal|fix|manage)\b",
        r"\b(medical\s*advice|treatment\s*plan|home\s*remed(y|ies)|clinical\s*decision)\b",
        r"\bhow\s*to\s*(perform|do)\s*(cpr|first\s*aid|surgery)\b",
        r"\b(should\s*i\s*(take\s*my\s*kid|go\s*to\s*the\s*doctor\s*for\s*my\s*fever))\b",
        r"\b(managing|manage)\s*(diabetes|hypertension|asthma|migraine)\b",
    ]

    def check_scope(self, text: str) -> SafetyCheckResult:
        """
        Evaluates whether a user query is within the operational ER scope or
        violates medical safety boundaries.

        Returns:
            SafetyCheckResult(is_safe=True) for operational questions.
            SafetyCheckResult(is_safe=False, refusal_message=...) for clinical/medical queries.
        """
        if not text or not text.strip():
            return SafetyCheckResult(is_safe=True)

        cleaned = text.lower().strip()

        # Check Diagnosis / Symptoms
        for pat in self.DIAGNOSIS_PATTERNS:
            if re.search(pat, cleaned):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="Medical diagnosis/symptom inquiry detected.",
                    refusal_message=self.SAFETY_REFUSAL_MESSAGE,
                )

        # Check Medication / Prescriptions
        for pat in self.MEDICATION_PATTERNS:
            if re.search(pat, cleaned):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="Medication/prescription advice inquiry detected.",
                    refusal_message=self.SAFETY_REFUSAL_MESSAGE,
                )

        # Check Medical Treatment / Clinical Intervention
        for pat in self.TREATMENT_PATTERNS:
            if re.search(pat, cleaned):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="Medical treatment/clinical procedure inquiry detected.",
                    refusal_message=self.SAFETY_REFUSAL_MESSAGE,
                )

        return SafetyCheckResult(is_safe=True)

    def is_within_scope(self, text: str) -> bool:
        """Helper boolean check."""
        return self.check_scope(text).is_safe


# Global singleton instance
safety_guard = SafetyGuard()
