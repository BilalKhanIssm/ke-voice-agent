import hashlib
import json
import re
from typing import Annotated

from livekit.agents import llm
from pydantic import Field

from app.core.transcript_utils import format_ke_reference_spoken_en, format_ke_reference_spoken_urdu


def _compact_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _mock_outage_payload(area_or_account: str) -> dict[str, object]:
    """Deterministic mock outage snapshot for demos until a live CMS/OMS API is wired."""
    key = (area_or_account or "unknown").strip().lower()
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    fault_types = (
        ("rain_related_trip", "Rain and moisture caused feeder protection to operate (trip)."),
        ("cable_fault", "An underground cable fault was detected on the network."),
        ("overhead_fault", "Weather or vegetation caused a fault on an overhead section."),
        ("substation_equipment", "Substation equipment required isolation for safe inspection."),
    )
    fault_code, fault_desc = fault_types[h % len(fault_types)]
    feeders = ("PECHS-07", "DHA-South-03", "North-Karachi-12", "Gulshan-05", "SITE-02")
    feeder = feeders[h % len(feeders)]
    ref = f"KE-{h % 10_000_000_000:010d}"
    return {
        "ok": True,
        "lookup": key,
        "feeder_name": feeder,
        "feeder_status": "out_of_service" if h % 3 != 0 else "partially_restored",
        "fault_type": fault_code,
        "fault_summary": fault_desc,
        "affected_scope": "Multiple households and adjacent blocks on the same feeder segment are affected.",
        "crew_status": "A technical team is on site working to restore supply safely.",
        "delay_factors": (
            "Standing water and safety conditions can slow work until the area is safe to energise."
            if fault_code == "rain_related_trip"
            else "Isolation and testing steps are being completed before re-energisation."
        ),
        "eta_summary": "Restoration is expected as soon as conditions and testing allow — often within a few hours, subject to field updates.",
        "complaint_reference": ref,
        "complaint_reference_spoken_ur": format_ke_reference_spoken_urdu(ref),
        "complaint_reference_spoken_en": format_ke_reference_spoken_en(ref),
        "complaint_already_logged": True,
        "priority_message": "The event is logged and being handled on priority; major updates are communicated when available.",
    }


def _mock_complaint_reference_payload(area_or_account: str) -> dict[str, object]:
    """Standalone mock complaint reference for demos (same seed as outage mock for a given key)."""
    raw = (area_or_account or "").strip()
    key = raw.lower() if raw else "unknown"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    ref = f"KE-{h % 10_000_000_000:010d}"
    return {
        "mock": True,
        "complaint_reference": ref,
        "complaint_reference_spoken_ur": format_ke_reference_spoken_urdu(ref),
        "complaint_reference_spoken_en": format_ke_reference_spoken_en(ref),
        "note": "Demo-only reference; not connected to a live CRM.",
    }


class LlmTools:
    @llm.function_tool(
        description=(
            "Fetch current outage / feeder status for this caller. "
            "Use when they give a neighbourhood with optional block/scheme (e.g. Gulistan-e-Johar block 18, جوہر بلاک 18) "
            "or a 13-digit KE account. Neighbourhood+block is enough for outage/cause — do not require account only for that. "
            "Never invent live field status — always call this before stating feeder state, fault type, crew activity, or ETA."
        )
    )
    async def get_outage_status(
        self,
        area_or_account: Annotated[
            str,
            Field(
                description=(
                    "Neighbourhood + block/landmark in the caller’s words, or 13-digit KE account digits only. "
                    "Prefer the same Urdu/Roman phrase they used (e.g. جوہر بلاک 18). Do not merge broken digit groups."
                )
            ),
        ],
    ) -> str:
        raw = (area_or_account or "").strip()
        digits = _compact_digits(raw)
        payload = _mock_outage_payload(raw if raw else digits)
        payload["account_digits_received"] = bool(len(digits) >= 10)
        return json.dumps(payload)

    @llm.function_tool(
        description=(
            "Return a demo complaint / service reference number for this caller (mock data only). "
            "Call when the customer asks for a complaint number, reference number, or 'شکایت نمبر' / 'ریفرنس نمبر'. "
            "Pass the same area, block, landmark, or 13-digit account string you are using from the conversation. "
            "In Urdu replies, read complaint_reference_spoken_ur verbatim when speaking the code; in English use "
            "complaint_reference_spoken_en. Do not imply a live national CRM ticket unless integrated."
        )
    )
    async def get_complaint_reference(
        self,
        area_or_account: Annotated[
            str,
            Field(
                description=(
                    "Area, block, scheme, landmark, or 13-digit KE account as given in the conversation "
                    "(same key you would pass to get_outage_status)."
                )
            ),
        ],
    ) -> str:
        return json.dumps(_mock_complaint_reference_payload(area_or_account))
