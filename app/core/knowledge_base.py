"""In-memory K-Electric knowledge base with lightweight keyword retrieval.
 
DATA SOURCES (last verified April 2026):
  - ke.com.pk product/service pages (crawled/searched Apr 2026)
  - NEPRA Prosumer Regulations 2026 (net billing framework, effective Feb 8 2026)
  - KE Live app listing (Google Play, Jan 2026)
  - KE Schedule of Charges & Tariff structure FY 2025-26
  - Public helpline / SMS / WhatsApp service documentation
"""
 
from __future__ import annotations
 
import re
from dataclasses import dataclass, field
from typing import Literal
 
 
@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    category: str
    keywords: tuple[str, ...]
    content: str
    urdu_keywords: tuple[str, ...] = field(default_factory=tuple)
 
 
_ENTRIES: tuple[KnowledgeEntry, ...] = (
 
    # ── Contact / Helpline ──────────────────────────────────────────────────
    KnowledgeEntry(
        id="helpline",
        category="contact",
        keywords=(
            "helpline", "contact", "phone number", "call center",
            "customer service", "support", "contact number", "complaint number",
            "how to call", "ke number", "whatsapp", "email ke",
            "customer care", "ke helpline", "118",
        ),
        content=(
            "K-Electric 24/7 Helpline: dial 118 (from any landline or mobile in Pakistan). "
            "Alternate number: (021) 99000. "
            "International / out-of-Pakistan: +92-21-3263-7133 or +92-21-3870-9132. "
            "WhatsApp: save 0348-0000118 and send 'Hi' to start. "
            "Email (general): customer.care@ke.com.pk. "
            "Email (app issues): ke@ke.com.pk. "
            "Head Office: KE House, 39-B Sunset Boulevard, Phase-II, DHA, Karachi. "
            "Social media support (24/7): Facebook facebook.com/kelectricpk | X (Twitter) @KElectricPk."
        ),
        urdu_keywords=("ہیلپ لائن", "نمبر", "رابطہ", "شکایت نمبر", "بجلی نمبر", "واٹس ایپ", "کے الیکٹرک نمبر"),
    ),
    KnowledgeEntry(
        id="ibc_centers",
        category="contact",
        keywords=(
            "ibc", "customer care center", "service center", "ke office",
            "integrated business center", "walk in", "branch office",
            "visit ke", "nearest center", "ke center location",
        ),
        content=(
            "K-Electric Integrated Business Centers (IBCs) are walk-in customer service centers "
            "spread across Karachi. IBCs handle: new connections, name/ownership changes, "
            "meter complaints, billing disputes, and document submission. "
            "Your designated IBC is printed on your KE bill under 'Reach K-Electric'. "
            "Use the KE Live app → 'Find Your Nearest Customer Care Centre' to locate IBCs by area."
        ),
        urdu_keywords=("آئی بی سی", "سروس سینٹر", "دفتر", "کے الیکٹرک آفس", "قریب ترین سینٹر"),
    ),
    KnowledgeEntry(
        id="complaint_process",
        category="contact",
        keywords=(
            "complaint", "grievance", "escalate", "feedback", "dispute",
            "report issue", "unresolved", "complaint process", "raise complaint",
            "lodge complaint", "file complaint", "how to complain",
        ),
        content=(
            "Lodge a complaint via: "
            "(1) Helpline 118 — agent assigns a unique 10-digit complaint number for follow-up. "
            "(2) KE Live app → 'Lodge Complaints' (electricity or billing). "
            "(3) WhatsApp 0348-0000118. "
            "(4) SMS: type COMP [space] [13-digit A/C #] and send to 8119. "
            "(5) Visit nearest IBC with your account number. "
            "(6) Email customer.care@ke.com.pk. "
            "Unresolved issues can be escalated to NEPRA's Consumer Complaint Portal at nepra.org.pk."
        ),
        urdu_keywords=("شکایت", "مسئلہ", "شکایت درج کریں", "شکایت کیسے کریں", "کمپلین"),
    ),
 
    # ── Billing ─────────────────────────────────────────────────────────────
    KnowledgeEntry(
        id="duplicate_bill",
        category="billing",
        keywords=(
            "duplicate bill", "download bill", "get bill", "lost bill",
            "bill copy", "ke bill online", "view bill", "online bill",
            "bill not received", "bill download", "e-bill",
        ),
        content=(
            "Download your KE duplicate bill instantly via: "
            "(1) KE Live app → Billing Details & History. "
            "(2) ke.com.pk → 'View & Pay Bill' using your 13-digit account number. "
            "(3) WhatsApp 0348-0000118 → request duplicate bill. "
            "(4) SMS: type BILL [space] [13-digit A/C #] and send to 8119. "
            "(5) Visit any IBC or call 118. "
            "No charges for duplicate bill download via digital channels."
        ),
        urdu_keywords=("ڈپلیکیٹ بل", "بل ڈاؤنلوڈ", "بل نہیں ملا", "آن لائن بل", "بجلی بل"),
    ),
    KnowledgeEntry(
        id="bill_payment",
        category="billing",
        keywords=(
            "pay bill", "bill payment", "how to pay", "bill due date",
            "payment method", "online payment", "jazzcash", "easypaisa",
            "bank payment", "late payment", "surcharge", "missed due date",
        ),
        content=(
            "Pay your KE bill via: "
            "KE Live app (debit/credit card), ke.com.pk online portal, "
            "JazzCash, EasyPaisa, bank mobile apps, ATMs, or any bank branch. "
            "WhatsApp 0348-0000118 also supports bill payment. "
            "Late payment surcharge: 10% of the billed amount (excluding government taxes/duties) "
            "if payment is not made by the due date. "
            "Tip: after paying online, verify the payment is reflected in your account before the due date."
        ),
        urdu_keywords=("بل ادائیگی", "بل کیسے جمع کریں", "لیٹ فیس", "دیر سے ادائیگی", "جمع بل"),
    ),
    KnowledgeEntry(
        id="high_bill_complaint",
        category="billing",
        keywords=(
            "high bill", "excessive bill", "overbilling", "bill too high",
            "wrong bill", "overcharged", "inflated bill", "incorrect bill",
            "estimation bill", "estimated reading", "detection bill",
        ),
        content=(
            "If you receive a high, estimated, or incorrect bill: "
            "(1) Call 118 or visit your IBC with your bill and CNIC to raise a billing complaint. "
            "(2) Request a meter inspection / re-reading if you believe the reading is wrong. "
            "(3) Estimated bills are issued when the meter cannot be read — an actual reading adjusts future bills. "
            "(4) Detection bills relate to meter tampering investigations — dispute at IBC with your CNIC. "
            "NEPRA tariff structure is available at ke.com.pk/tariff-structure for reference."
        ),
        urdu_keywords=("زیادہ بل", "غلط بل", "اندازہ بل", "بل کی شکایت", "ڈیٹیکشن بل"),
    ),
    KnowledgeEntry(
        id="account_number",
        category="billing",
        keywords=(
            "account number", "ke account number", "reference number",
            "consumer number", "contract number", "find account number",
            "13 digit", "where is account number",
        ),
        content=(
            "Your KE Account Number is a 13-digit number printed prominently on your electricity bill "
            "under 'Consumer Details'. It is required for all interactions with K-Electric "
            "(helpline, app, WhatsApp, SMS service). "
            "Note: your bill also shows a Contract Number and Consumer Number — "
            "the Account Number is the primary identifier for customer service."
        ),
        urdu_keywords=("اکاؤنٹ نمبر", "ریفرنس نمبر", "13 ہندسہ", "بل نمبر"),
    ),
    KnowledgeEntry(
        id="income_tax_certificate",
        category="billing",
        keywords=(
            "income tax certificate", "tax certificate", "tax filer",
            "fbr tax", "non filer surcharge", "filer exemption", "tax deduction",
            "7.5 percent tax",
        ),
        content=(
            "Non-filers (not on FBR Active Taxpayer List) are charged an additional 7.5% tax "
            "on electricity bills above Rs. 25,000 (effective July 1, 2021). "
            "To claim exemption: update your FBR taxpayer status via ke.com.pk (FBR link provided on site). "
            "Income Tax Certificate: available via WhatsApp 0348-0000118 or ke.com.pk. "
            "Call 118 or visit IBC for assistance."
        ),
        urdu_keywords=("انکم ٹیکس سرٹیفکیٹ", "ٹیکس فائلر", "ایف بی آر", "ٹیکس چھوٹ", "نان فائلر"),
    ),
 
    # ── Tariff ──────────────────────────────────────────────────────────────
    KnowledgeEntry(
        id="tariff_structure",
        category="billing",
        keywords=(
            "tariff", "electricity rate", "per unit rate", "kw rate",
            "lifeline tariff", "residential tariff", "commercial tariff",
            "peak hours", "off peak", "unit price", "kwh rate", "nepra tariff",
        ),
        content=(
            "KE tariffs are set by NEPRA. Key FY 2025-26 rates (residential, >100 units): "
            "base average tariff ~PKR 31.59/kWh. "
            "Lifeline (0-50 units): PKR 3.95/kWh. "
            "51-100 units: PKR 7.74/kWh. "
            "Peak hours (residential): 6:30 PM – 10:30 PM (Apr-Oct); 6:00 PM – 10:00 PM (Nov-Mar). "
            "Higher tariff applies during peak hours for time-of-use consumers. "
            "Full tariff schedule: ke.com.pk/tariff-structure."
        ),
        urdu_keywords=("ٹیرف", "فی یونٹ قیمت", "بجلی کی قیمت", "پیک اوقات", "لائف لائن"),
    ),
 
    # ── Power Outage / Load Shedding ─────────────────────────────────────────
    KnowledgeEntry(
        id="power_outage",
        category="outage",
        keywords=(
            "power outage", "no electricity", "light gone", "bijli nahi",
            "power failure", "electricity off", "blackout", "no power",
            "power cut", "power not restored", "trip", "tripped",
        ),
        content=(
            "Report a power outage: call 118 (24/7), use KE Live app → 'Power Status' or 'Lodge Complaint', "
            "WhatsApp 0348-0000118, or SMS: COMP [space] [13-digit A/C #] to 8119. "
            "Check live power status for your area via KE Live app → 'Live Power Status'. "
            "Provide your account number and address for faster resolution. "
            "For safety hazards (fallen wires, sparking poles), call 118 immediately — available 24/7."
        ),
        urdu_keywords=("بجلی نہیں", "لائٹ گئی", "بجلی بند", "آؤٹیج", "بجلی کی شکایت"),
    ),
    KnowledgeEntry(
        id="load_shedding",
        category="outage",
        keywords=(
            "load shedding", "loadshedding", "load shed schedule", "load shedding schedule",
            "loadshedding timing", "load shedding area", "scheduled outage",
            "unscheduled outage", "how many hours", "load shedding hours",
        ),
        content=(
            "Check your area's load shedding schedule via: "
            "(1) KE Live app → 'Load-shed Schedules'. "
            "(2) ke.com.pk → Load Shedding Schedule. "
            "(3) SMS: type LS [space] [13-digit A/C #] and send to 8119. "
            "(4) WhatsApp 0348-0000118. "
            "Load shedding occurs when electricity demand exceeds available supply. "
            "KE publishes scheduled outage timings online; unscheduled outages are due to faults or emergencies. "
            "For unscheduled outages longer than the scheduled period, lodge a complaint via 118 or KE Live."
        ),
        urdu_keywords=("لوڈ شیڈنگ", "بجلی کب آئے گی", "شیڈول", "کتنے گھنٹے بجلی بند"),
    ),
    KnowledgeEntry(
        id="safety_hazard",
        category="outage",
        keywords=(
            "safety hazard", "fallen wire", "sparking wire", "electric shock",
            "dangerous pole", "wire on ground", "electric fire", "emergency",
            "live wire", "transformer fire",
        ),
        content=(
            "For any electrical safety emergency — fallen wires, sparking equipment, "
            "transformer fires, or electric shocks — call 118 immediately (24/7 emergency line). "
            "You can also report via KE Live app → 'Report Safety Hazard'. "
            "Do NOT touch or approach fallen electrical wires. Keep a safe distance and alert others."
        ),
        urdu_keywords=("حفاظتی خطرہ", "تار گرا", "بجلی کا کرنٹ", "خطرناک", "ہنگامی صورتحال"),
    ),
 
    # ── New Connection ───────────────────────────────────────────────────────
    KnowledgeEntry(
        id="new_connection",
        category="connection",
        keywords=(
            "new connection", "new meter", "apply connection", "electricity connection",
            "new electricity", "apply for meter", "how to get connection",
            "residential connection", "commercial connection",
        ),
        content=(
            "Apply for a new K-Electric connection: "
            "(1) Online at ke.com.pk → Customer Services → New Connection. "
            "(2) Visit your nearest IBC. "
            "Documents required: CNIC copy, property ownership/tenancy documents, "
            "load details (appliances and wattage), and site plan. "
            "Process: submit application → KE surveys premises → demand notice issued → "
            "pay demand notice → meter installed. "
            "Call 118 or visit IBC for queries."
        ),
        urdu_keywords=("نیا کنکشن", "نیا میٹر", "بجلی کنکشن", "کنکشن کیسے لیں", "میٹر لگوانا"),
    ),
    KnowledgeEntry(
        id="name_change_ownership",
        category="connection",
        keywords=(
            "name change", "ownership change", "change name", "transfer connection",
            "owner change", "new owner", "tenancy change", "change of consumer",
        ),
        content=(
            "To change the name or ownership of a KE connection: visit your designated IBC with: "
            "CNIC of new owner, sale/transfer deed or tenancy agreement, "
            "latest paid electricity bill, and an application form (available at IBC). "
            "The IBC printed on your bill is your designated service center. "
            "Call 118 for guidance on required documents."
        ),
        urdu_keywords=("نام تبدیل", "ملکیت تبدیل", "نام بدلنا", "نیا مالک", "کنکشن ٹرانسفر"),
    ),
    KnowledgeEntry(
        id="load_change",
        category="connection",
        keywords=(
            "load change", "increase load", "reduce load", "sanctioned load",
            "phase change", "single phase", "three phase", "3 phase", "1 phase",
            "load extension", "load reduction",
        ),
        content=(
            "To increase, reduce, or change sanctioned load (e.g., single-phase to three-phase): "
            "visit your designated IBC with CNIC, latest electricity bill, and load details. "
            "A KE survey will assess feasibility and a demand notice will be issued if approved. "
            "Note: your connected load must match the sanctioned load — mismatches can affect "
            "meter inspections and solar/net metering applications. Call 118 for information."
        ),
        urdu_keywords=("لوڈ تبدیل", "لوڈ بڑھانا", "فیز تبدیل", "سنگل فیز", "تھری فیز"),
    ),
 
    # ── Meter Issues ─────────────────────────────────────────────────────────
    KnowledgeEntry(
        id="meter_complaint",
        category="meter",
        keywords=(
            "meter complaint", "faulty meter", "meter not working", "meter reading",
            "wrong meter reading", "meter problem", "meter fast", "meter running fast",
            "running too fast", "meter change", "meter replacement", "burnt meter",
            "damaged meter", "meter slow", "meter defective",
        ),
        content=(
            "Report a faulty, damaged, or fast-running meter: "
            "(1) Call 118 — a complaint is registered with a reference number. "
            "(2) KE Live app → Lodge Complaint → Electricity Complaint. "
            "(3) Visit IBC with your CNIC and latest bill. "
            "KE will dispatch a team to inspect and replace the meter if faulty. "
            "Meter replacement for verified faults is done at no charge to the consumer."
        ),
        urdu_keywords=("میٹر شکایت", "خراب میٹر", "میٹر تیز", "میٹر تبدیل", "میٹر مسئلہ"),
    ),
    KnowledgeEntry(
        id="meter_reading",
        category="meter",
        keywords=(
            "meter reading", "read meter", "meter units", "how to read meter",
            "meter reading schedule", "when is meter read", "meter reader",
        ),
        content=(
            "KE reads meters monthly. Your meter reading schedule can be found at ke.com.pk "
            "or via KE Live app. If the meter could not be read (inaccessible premises), "
            "KE issues an estimated bill — the actual difference is adjusted on the next bill. "
            "Ensure your meter is accessible to the meter reader to avoid estimated bills. "
            "Dispute a reading by calling 118 or visiting your IBC."
        ),
        urdu_keywords=("میٹر ریڈنگ", "یونٹ ریڈنگ", "میٹر شیڈول", "اندازہ بل"),
    ),
 
    # ── Digital Services ─────────────────────────────────────────────────────
    KnowledgeEntry(
        id="ke_live_app",
        category="digital",
        keywords=(
            "ke live", "ke app", "mobile app", "app download",
            "digital services", "online services", "ke live app",
            "app features", "ke application",
        ),
        content=(
            "KE Live — K-Electric's official mobile app (iOS & Android): "
            "Live power status | Load shedding schedules | Billing details & history | "
            "Bill payment | Real-time KE notifications | Find nearest IBC | "
            "Report power theft | Report safety hazard | Lodge electricity & billing complaints | "
            "Complaint history tracking. "
            "Download: search 'KE Live' on Google Play Store or Apple App Store. "
            "Support: email ke@ke.com.pk."
        ),
        urdu_keywords=("کے ای لائیو", "کے ای ایپ", "موبائل ایپ", "آن لائن سروس"),
    ),
    KnowledgeEntry(
        id="sms_service",
        category="digital",
        keywords=(
            "sms service", "8119", "ke sms", "text service", "register sms",
            "sms alert", "sms bill", "sms complaint", "sms load shedding",
        ),
        content=(
            "K-Electric SMS Service (send to 8119): "
            "Register: REG [space] [13-digit A/C #] → confirmation SMS received. "
            "Check bill: BILL [space] [13-digit A/C #]. "
            "Lodge complaint: COMP [space] [13-digit A/C #]. "
            "Load shedding schedule: LS [space] [13-digit A/C #]. "
            "General chat/query: CHAT [space] your message. "
            "Unsubscribe: UNREG [space] [13-digit A/C #]. "
            "Also register online at ke.com.pk with your account number and mobile number."
        ),
        urdu_keywords=("ایس ایم ایس سروس", "8119", "ٹیکسٹ سروس", "ایس ایم ایس رجسٹریشن"),
    ),
    KnowledgeEntry(
        id="whatsapp_service",
        category="digital",
        keywords=(
            "whatsapp ke", "whatsapp service", "ke whatsapp",
            "whatsapp bill", "whatsapp complaint", "whatsapp number ke",
        ),
        content=(
            "K-Electric WhatsApp Service: save 0348-0000118 and send 'Hi' to begin. "
            "Available 24/7. Services: duplicate bill download, billing complaints, "
            "technical complaints, income tax certificate, load shedding schedule, "
            "new connection information, and bill payment. "
            "Fastest digital channel for self-service without calling 118."
        ),
        urdu_keywords=("واٹس ایپ", "کے ای واٹس ایپ", "واٹس ایپ بل", "واٹس ایپ شکایت"),
    ),
 
    # ── Solar / Net Metering ─────────────────────────────────────────────────
    KnowledgeEntry(
        id="net_metering",
        category="solar",
        keywords=(
            "net metering", "solar", "solar panel", "solar connection",
            "export electricity", "solar billing", "net billing",
            "bidirectional meter", "bi directional meter", "solar ke",
            "solar application", "solar approval", "rooftop solar",
        ),
        content=(
            "K-Electric Net Metering / Net Billing (2026 update): "
            "NEW applications (after Feb 8 2026) fall under NEPRA Prosumer Regulations 2026 "
            "— net billing framework: imports charged at retail tariff; "
            "exports credited at ~PKR 13/kWh (national average purchase price). "
            "Applications submitted before Feb 8 2026 are processed under old net metering rules "
            "(one-to-one unit offset at ~PKR 27/kWh) as per Power Ministry directive. "
            "Existing agreements continue until expiry. "
            "Requirements: AEDB-certified vendor (cannot apply directly), "
            "on-grid or hybrid system (off-grid not eligible), "
            "connected load matching sanctioned load, usually 3-phase meter. "
            "Apply at ke.com.pk/net-metering/ or visit Net Metering Facilitation Centre, Elandar Complex. "
            "Approval: 60-90 days. KE installs bi-directional meter after approval."
        ),
        urdu_keywords=("نیٹ میٹرنگ", "سولر", "سولر پینل", "سولر کنکشن", "نیٹ بلنگ", "ایکسپورٹ بجلی"),
    ),
    KnowledgeEntry(
        id="power_theft_report",
        category="services",
        keywords=(
            "power theft", "electricity theft", "kunda", "illegal connection",
            "report theft", "hook", "bijli chori", "meter tampering",
            "meter bypass",
        ),
        content=(
            "Report power theft or illegal connections (kundas): "
            "KE Live app → 'Speak Up – Report Power Theft'. "
            "Call 118 or WhatsApp 0348-0000118. "
            "Complaints can also be submitted via Facebook (facebook.com/kelectricpk). "
            "Reports are confidential. Power theft causes increased load shedding for all consumers."
        ),
        urdu_keywords=("بجلی چوری", "کنڈا", "غیر قانونی کنکشن", "میٹر خرابی", "چوری رپورٹ"),
    ),
 
    # ── NEPRA Escalation ─────────────────────────────────────────────────────
    KnowledgeEntry(
        id="nepra_escalation",
        category="regulatory",
        keywords=(
            "nepra", "escalate", "regulator", "unresolved complaint",
            "nepra complaint", "consumer rights", "nepra portal",
        ),
        content=(
            "If your complaint remains unresolved after contacting K-Electric, "
            "escalate to NEPRA (National Electric Power Regulatory Authority): "
            "File online at nepra.org.pk (Consumer Complaint portal). "
            "NEPRA is the federal electricity regulator that oversees K-Electric's service standards. "
            "Keep your KE complaint number (received when you first filed with KE) for reference."
        ),
        urdu_keywords=("نیپرا", "ریگولیٹر", "حل نہ ہونے والی شکایت", "نیپرا پورٹل"),
    ),
)
 
 
# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------
_TOOL_SIGNALS: frozenset[str] = frozenset({
    "power status", "live status", "bijli hai", "current status",
    "outage now", "light hai", "bijli kab", "is there electricity",
    "outage status", "feeder status", "feeder", "no light", "power outage",
    "when will power", "restoration", "supply status", "mera ilaqa",
    "check status", "status check", "update dein", "any update",
    "tripped", "fault", "crew", "site par", "field team",
})
 
_MIXED_SIGNALS: frozenset[str] = frozenset({
    "no electricity", "bijli nahi", "light nahi", "power cut",
    "load shedding area", "outage complaint", "bill not correct",
    "meter problem complaint", "faulty meter complaint",
    "complaint status", "complaint number", "complaint update",
})
 
_TRIVIAL_SIGNALS: frozenset[str] = frozenset({
    "hello", "hi ", " hi", "assalam", "salam", "aoa", "bye", "goodbye",
    "khuda hafiz", "thank you", "thanks", "shukriya", "ok ", " ok", "okay",
    "alright", "sure", "haan", "nahi", "theek", "bilkul",
})

# Goodbye / wrap-up: used to skip tool/RAG nudges and force a short sign-off.
_STRONG_CLOSING_ROMAN: frozenset[str] = frozenset({
    "allah hafiz",
    "khuda hafiz",
    "goodbye",
    "good bye",
    "bye bye",
    "alvida",
    "fi amanillah",
    "fee aman",
})
_AR_CLOSING_PHRASES: tuple[str, ...] = (
    "اللہ حافظ",
    "خدا حافظ",
    "الوداع",
    "فی امان اللہ",
)


def is_closing_utterance(text: str) -> bool:
    """
    True when the caller is clearly ending the call (goodbye / thanks on hang-up).

    Kept conservative: strong goodbye phrases, or short thanks tied to leaving/call.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    for phrase in _AR_CLOSING_PHRASES:
        if phrase in raw:
            return len(raw.split()) <= 20
    t = normalize_query_for_retrieval(text)
    low = t.lower()
    if any(p in low for p in _STRONG_CLOSING_ROMAN):
        return len(low.split()) <= 20
    words = low.split()
    if len(words) <= 4 and low.strip() in {"bye", "goodbye", "khuda hafiz", "allah hafiz", "alvida"}:
        return True
    if len(words) <= 8 and any(x in low for x in ("thank you", "thanks", "shukriya", "شکری")):
        if any(x in low for x in ("call", "calle", "bye", "allah", "khuda", "goodbye", "کال")):
            return True
    return False


def classify_intent(
    text: str,
) -> Literal["tool", "rag", "mixed", "none"]:
    """
    Route user query to: tool (real-time data), rag (informational),
    mixed (needs both), or none (greetings / trivial).

    Rule-based, runs in <1 ms.
    """
    t = normalize_query_for_retrieval(text)
    words = t.split()

    # Thirteen+ consecutive digits (after stripping non-digits): typical KE account readout.
    if len(re.sub(r"\D", "", text)) >= 13:
        return "tool"

    if len(words) <= 4 and (
        t.strip() in {"hi", "hey", "yo"}
        or any(sig in t for sig in _TRIVIAL_SIGNALS)
    ):
        return "none"
 
    if any(sig in t for sig in _MIXED_SIGNALS):
        return "mixed"
 
    if any(sig in t for sig in _TOOL_SIGNALS):
        return "tool"
 
    return "rag"
 
 
# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACED_ACRONYM_RE = re.compile(r"\b(?:[a-zA-Z]\s+){2,}[a-zA-Z]\b")
_MULTISPACE_RE = re.compile(r"\s+")
 
 
def normalize_query_for_retrieval(text: str) -> str:
    """
    Normalize user text for robust keyword retrieval.
    Handles spaced acronyms, STT variants, common KE-specific spellings,
    and punctuation.
    """
    lowered = text.lower()
 
    def _collapse_spaced(m: re.Match[str]) -> str:
        return m.group(0).replace(" ", "")
 
    lowered = _SPACED_ACRONYM_RE.sub(_collapse_spaced, lowered)
    lowered = _PUNCT_RE.sub(" ", lowered)
    lowered = _MULTISPACE_RE.sub(" ", lowered).strip()
 
    # Phonetic / STT / common misspelling variants
    lowered = re.sub(r"\bk electric\b", "k-electric", lowered)
    lowered = re.sub(r"\bkesc\b", "k-electric", lowered)
    lowered = re.sub(r"\bkarachi electric\b", "k-electric", lowered)
    lowered = re.sub(r"\bloadshedding\b", "load shedding", lowered)
    lowered = re.sub(r"\bload shed\b", "load shedding", lowered)
    lowered = re.sub(r"\bijara\b", "ijarah", lowered)
    lowered = re.sub(r"\bbijli\b", "electricity", lowered)
    lowered = re.sub(r"\bbill amount\b", "bill", lowered)
    lowered = re.sub(r"\bnepra\b", "nepra", lowered)
    lowered = re.sub(r"\bibc\b", "integrated business center", lowered)
    lowered = re.sub(r"\bke live\b", "ke live app", lowered)
    lowered = re.sub(r"\bnet meter\b", "net metering", lowered)
    lowered = re.sub(r"\bsolar meter\b", "net metering", lowered)
 
    return lowered
 
 
def _score_entry(entry: KnowledgeEntry, query_lower: str, is_urdu: bool) -> int:
    score = sum(1 for kw in entry.keywords if kw in query_lower)
    if is_urdu:
        score += sum(2 for kw in entry.urdu_keywords if kw in query_lower)
    return score
 
 
def retrieve(
    query: str,
    language: Literal["en", "ur"] = "en",
    top_k: int = 2,
) -> list[KnowledgeEntry]:
    """
    Return up to top_k most relevant entries for the query.
    Uses keyword overlap scoring — O(entries * keywords), typically <1 ms.
    Returns empty list if no entry scores above zero.
    """
    query_lower = normalize_query_for_retrieval(query)
    is_urdu = language == "ur"
 
    scored = [
        (score, entry)
        for entry in _ENTRIES
        if (score := _score_entry(entry, query_lower, is_urdu)) > 0
    ]
    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:top_k]]