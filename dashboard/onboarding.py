"""AI-powered interactive onboarding guide -- a mentor-style walkthrough
layered ON TOP of the existing analyst dashboard. Pure UI/UX: this module
never touches models, the training pipeline, datasets, inference, or any
backend/API -- it only reads `st.session_state`, a small local JSON
progress file, and data the rest of `dashboard/app.py` already loaded.

Design constraints this module works within (both genuine Streamlit
platform limits, not corners cut for convenience):

- `st.tabs` exposes no "which tab is currently selected" API to Python, and
  there is no first-class way to switch tabs from server-side code. This
  module drives tab changes by injecting a small, defensive JS snippet
  (via `st.components.v1.html`) that finds the real tab button in the
  parent document and clicks it only if it isn't already selected -- the
  same "reach into the real DOM" technique Streamlit's own community uses
  for this exact gap. Deliberately re-asserted on EVERY render while the
  tour is active (`_ensure_active_tab`), not a one-shot "queue a switch,
  consume it next render" flag -- an earlier one-shot version turned out
  fragile in practice (a queued switch that didn't get consumed on the
  very next rerun, e.g. one triggered from inside an `@st.dialog`
  callback, would fire late on some unrelated later click instead, which
  is what caused "Tell me more" to visibly jump to a stale target page).
  The `aria-selected` check before clicking makes repeated calls a true
  no-op once already on the right tab, so re-running it every time is safe
  and never fights a user's own manual clicks once they've exited the tour.
- Because there is no reliable signal for "the user manually changed tabs
  while the tour was mid-page," this module approximates context-awareness
  (requirement #9) by scoping all tour chrome (spotlight/dim, "resume?"
  prompts) to the page the CURRENT step targets: if a user free-navigates
  away, the tour's highlight/dim simply doesn't render on the page they're
  now looking at (no stale highlighting bleeding onto the wrong tab), and a
  "Resume walkthrough?" banner reappears the moment they land back on a
  page the tour cares about.
- "Persist between sessions" here means a local JSON file next to the
  dashboard data (this app has no login/multi-tenant identity anywhere
  else, so there is no per-user identity to key persistence on) -- shared
  across whoever opens this dashboard instance, consistent with every
  other "single-operator tool" assumption already in this project.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ONBOARDING_STATE_PATH = Path(__file__).resolve().parent / ".onboarding_state.json"

# Must exactly match main()'s st.tabs(...) labels -- the JS tab-switch
# bridge matches on this literal visible text.
TAB_OVERVIEW = "Overview"
TAB_LIVE = "Live Detection"
TAB_INVESTIGATE = "Investigate"
TAB_CAMPAIGNS = "Campaigns"
TAB_GRAPH = "Knowledge Graph"
TAB_MITRE = "MITRE & Threats"
TAB_MODEL = "Model & Drift"
TAB_HEATMAP = "Risk Heatmap"


@dataclass
class OnboardingStep:
    id: str
    page: str  # which tab this step lives on (must match a TAB_* constant)
    title: str
    story: str  # mentor-voice narrative framing, shown above the explanation
    basic: str  # plain "what this is / why it matters"
    advanced: str  # deeper technical explanation, behind "Tell me more"
    is_page_intro: bool = False  # whole-page spotlight vs. one widget among several


TOUR_STEPS: list[OnboardingStep] = [
    OnboardingStep(
        id="overview", page=TAB_OVERVIEW, is_page_intro=True,
        title="The Overview -- your morning briefing",
        story=(
            "Imagine you've just started your shift as a SOC analyst. Before you touch a single alert, "
            "you need one question answered: **how bad is today, overall?** This page is built to answer "
            "exactly that in the first five seconds."
        ),
        basic=(
            "The Overview aggregates everything the system has detected right now -- active threats, "
            "high-severity alerts, which attack types are showing up, and how the model is performing. "
            "Analysts start here to triage: is today calm, or is something on fire?"
        ),
        advanced=(
            "Every number here is LIVE relative to your sidebar filters and detection threshold -- moving "
            "the threshold slider recomputes precision/recall/false-positives-per-day in real time from the "
            "same held-out test-split scores the six-criteria evaluation report uses, so what you see here "
            "is never a separately-computed or approximated figure. Look closely at the Attack Coverage "
            "cards: `lateral_movement` and `device_spoofing` have only a handful of labeled test events "
            "each, next to `brute_force`'s dozens -- a real class-imbalance problem baked into how rarely "
            "these attacks occur, not a dashboard artifact. It was investigated and measurably improved "
            "rather than left as an unexplained weak spot -- see Model & Drift's 'Tell me more' for the "
            "real before/after numbers."
        ),
    ),
    OnboardingStep(
        id="live_detection", page=TAB_LIVE, is_page_intro=True,
        title="Live Detection -- watching the pipeline work",
        story=(
            "An attacker has begun targeting an employee account. Our system continuously observes "
            "authentication events, scoring each one the instant it arrives. Let's watch that happen."
        ),
        basic=(
            "This feed replays real, already-scored events from the model's held-out evaluation set in "
            "timestamp order, so you experience detection as a live stream -- the same pipeline stages "
            "(feature engineering -> behavioral model -> classifier -> risk score -> MITRE mapping -> alert) "
            "light up as each event is scored."
        ),
        advanced=(
            "This is explicitly a REPLAY of a fixed historical test split, not a connection to live traffic "
            "-- this project is a synthetic-data benchmarking system, not a production sensor network, and "
            "that's stated here rather than left ambiguous. What you're watching is real: the same scores, "
            "same classifications, same thresholds used everywhere else in this dashboard, just experienced "
            "in time order instead of as a static table."
        ),
    ),
    OnboardingStep(
        id="investigate_intro", page=TAB_INVESTIGATE,
        title="An alert just fired. Let's investigate.",
        story=(
            "Here it is -- the account you were watching just tripped a detection. This is where an "
            "analyst's real work begins: not trusting the alert blindly, but understanding it."
        ),
        basic=(
            "This table lists every event currently above your detection threshold. Select any row to "
            "open a full investigation -- everything below is built around ONE selected event at a time, "
            "the way a real analyst works a single case before moving to the next."
        ),
        advanced=(
            "Sortable by any column, including the model's own anomaly score -- clicking a column header "
            "re-sorts client-side. The threshold that determines what appears here is the same slider in "
            "the sidebar; lowering it surfaces more (and riskier) borderline events for review."
        ),
    ),
    OnboardingStep(
        id="investigate_incident_timeline", page=TAB_INVESTIGATE,
        title="Step 2: What led up to this moment?",
        story=(
            "Before reading any AI verdict, an analyst reconstructs the sequence of events -- a single "
            "alert rarely tells the whole story on its own."
        ),
        basic=(
            "A reconstructed, minute-by-minute timeline of this user's own recent events -- logins, "
            "resource access, device or location changes -- ending in the model's detection and the "
            "resulting alert. This is the raw sequence, before any AI interpretation."
        ),
        advanced=(
            "Built by diffing consecutive real events for this user within the loaded evaluation window "
            "only (earlier training-period raw events aren't in the dashboard artifacts, stated here "
            "rather than silently showing an incomplete picture). A device or location change between "
            "consecutive events is flagged distinctly, since that's often the first visible sign of "
            "credential misuse."
        ),
    ),
    OnboardingStep(
        id="investigate_event_details", page=TAB_INVESTIGATE,
        title="Step 3: What actually happened?",
        story="Before trusting any AI verdict, an analyst reads the raw facts first.",
        basic=(
            "This is the unmodified event record the model scored -- timestamp, user, department, device, "
            "location, resource accessed. No interpretation yet, just ground truth."
        ),
        advanced=(
            "Every field shown here is exactly what fed the feature-engineering pipeline -- nothing is "
            "reformatted or summarized for this view, so if something here looks odd, it genuinely was "
            "odd in the source event, not an artifact of the dashboard."
        ),
    ),
    OnboardingStep(
        id="investigate_ai_summary", page=TAB_INVESTIGATE,
        title="Step 4: What does the AI think happened?",
        story=(
            "Now we bring in the machine's read on the situation -- a plain-language summary an analyst "
            "can absorb in seconds, before diving into the technical evidence."
        ),
        basic=(
            "A narrative summary of this event's risk: what was predicted, how confident the model is, "
            "and which behaviors deviated most from this user's own history."
        ),
        advanced=(
            "Generated by reasoning over the pipeline's own real outputs (score percentile, per-feature "
            "deviation from this user's rolling baseline) -- not a canned template keyed off the predicted "
            "class alone."
        ),
    ),
    OnboardingStep(
        id="investigate_behavior_comparison", page=TAB_INVESTIGATE,
        title="Step 5: How different is this from normal?",
        story=(
            "Every person has habits -- when they log in, from where, on what device. An attacker wearing "
            "someone else's credentials usually can't fake ALL of those habits at once."
        ),
        basic=(
            "Our AI learns every user's normal behavior over time. Here we're comparing today's activity "
            "against that historical baseline. Large deviations often indicate compromised credentials or "
            "insider threats."
        ),
        advanced=(
            "Each behavioral feature (login velocity, geo-distance from home, device-switch rate, "
            "failed-login ratio, and more) is compared in standard-deviation units against this user's own "
            "rolling EMA baseline -- a deviation of 2+ std. devs is flagged distinctly, because it means "
            "'unusual for THIS person,' not just 'unusual in general.'"
        ),
    ),
    OnboardingStep(
        id="investigate_explainability", page=TAB_INVESTIGATE,
        title="Step 6: Why did the AI reach this conclusion?",
        story=(
            "A verdict without a reason isn't useful to a SOC analyst -- you can't act on 'trust me.' This "
            "is where the model shows its work."
        ),
        basic=(
            "SHAP explains why the AI made this prediction -- which specific factors pushed the risk score "
            "up or down, and by how much."
        ),
        advanced=(
            "SHAP estimates the contribution of each behavioral/graph feature toward the model's anomaly "
            "score using additive feature attribution -- the waterfall chart shows the exact cumulative "
            "effect of the top contributors, computed once offline (exact SHAP) plus a lightweight "
            "streaming approximation shown alongside it, so the documented cost/accuracy tradeoff between "
            "the two is visible, not hidden."
        ),
    ),
    OnboardingStep(
        id="investigate_timeline", page=TAB_INVESTIGATE,
        title="Step 7: How has this user trended over time?",
        story=(
            "One incident is a snapshot. Here we zoom out to this user's longer-run numeric trend -- "
            "different from the event-by-event reconstruction two steps ago."
        ),
        basic=(
            "Rolling trend lines for this user's key behavioral features (failed-login ratio, geo-distance "
            "from home, EMA failure rate, peer-group deviation) across their full history, with the "
            "flagged event marked -- so you can see whether this was a sudden spike or a slow drift."
        ),
        advanced=(
            "Plotted from this user's actual train + test history in the loaded run -- if this account is "
            "new or cold-start, that's stated explicitly here rather than silently showing an empty chart."
        ),
    ),
    OnboardingStep(
        id="investigate_mitre", page=TAB_INVESTIGATE,
        title="Step 8: Where does this fit in a real attack?",
        story=(
            "Finally, we map this event to the MITRE ATT&CK framework -- the shared vocabulary security "
            "teams worldwide use to describe attacker behavior, so this finding is speakable outside this "
            "dashboard too."
        ),
        basic=(
            "Shows which MITRE ATT&CK tactic and technique this event's attack type maps to, if any -- "
            "context for how this fits into the broader landscape of known attacker behavior."
        ),
        advanced=(
            "Sourced from `configs/mitre_mapping.yaml`, a maintained mapping table (not inferred per-event) "
            "-- this dataset models 5 of the 10 MITRE Enterprise tactics, stated plainly rather than "
            "implying broader coverage than exists."
        ),
    ),
    OnboardingStep(
        id="investigate_recommendation", page=TAB_INVESTIGATE,
        title="Step 9: So what should an analyst actually DO?",
        story=(
            "Everything up to now has been understanding. This is where the platform earns its keep: "
            "turning that understanding into a concrete, defensible recommendation."
        ),
        basic=(
            "A ranked list of recommended response actions (e.g. force MFA, suspend session, escalate to "
            "human review), each backed by which specific evidence signals support it -- plus, just as "
            "important, which OTHER plausible actions were considered and why they were rejected."
        ),
        advanced=(
            "An action only ever gets recommended if at least 2 independent evidence categories corroborate "
            "it -- a single strong-but-lone signal (say, MITRE tactic alone) is never enough by itself, and "
            "rejected actions show the real reason they fell short of that bar. Confidence language is also "
            "grounded in this model's REAL backtested precision for the predicted class (from the six-"
            "criteria evaluation), not a fabricated per-event score -- so a rare class like "
            "`device_spoofing` gets an explicit reliability caveat instead of false certainty. See 'Tell me "
            "more' on the Model & Drift step for the full story behind why that caveat exists."
        ),
    ),
    OnboardingStep(
        id="investigate_risk_profile", page=TAB_INVESTIGATE,
        title="Always on: this user's risk profile",
        story=(
            "Unlike the numbered steps so far, this panel never goes away while you're investigating -- "
            "an analyst working a case wants this at-a-glance context in view the whole time, not just "
            "for one step."
        ),
        basic=(
            "A persistent side panel giving an at-a-glance read on this ONE user: current risk score, "
            "prior alerts, device trust, typical login hour, location, and overall behavioral drift -- all "
            "in one place, so you don't have to re-derive it from the steps above."
        ),
        advanced=(
            "Every indicator here is a visual chip or readout over a value already computed elsewhere on "
            "this page (the same baseline, prior-alert count, and deviation figures used in the numbered "
            "steps) -- nothing here is a new or separately-computed data source, it's a summary view of "
            "what you've already seen."
        ),
    ),
    OnboardingStep(
        id="campaigns", page=TAB_CAMPAIGNS, is_page_intro=True,
        title="Campaigns -- seeing the whole operation",
        story=(
            "A single flagged event is one moment. But attackers run CAMPAIGNS -- coordinated sequences "
            "of events toward one goal. An analyst who only ever looks at individual alerts misses that."
        ),
        basic=(
            "Groups flagged events by their real underlying attack campaign, so you can see an entire "
            "operation -- its severity, its span, everything it touched -- as one story instead of "
            "disconnected alerts."
        ),
        advanced=(
            "Grouped by the dataset's real `attack_id` (ground-truth campaign identity), not inferred or "
            "clustered after the fact -- each card reflects an actual coordinated attack the generator "
            "created."
        ),
    ),
    OnboardingStep(
        id="knowledge_graph", page=TAB_GRAPH, is_page_intro=True,
        title="The Knowledge Graph -- relationships, not just records",
        story=(
            "Some threats aren't visible in any single event -- they only show up in the SHAPE of how "
            "users, devices, and resources connect."
        ),
        basic=(
            "An interactive graph of users, devices, and their relationships. Click any node to inspect it "
            "-- unusual connections (a device shared across many unrelated users, for instance) are often "
            "the first sign of credential sharing or lateral movement."
        ),
        advanced=(
            "Built from the same bipartite user-device graph and per-department resource-transition graphs "
            "the feature-engineering pipeline maintains internally (`feature_engineering/graph.py`) -- this "
            "view visualizes real internal model state, not a separate illustrative diagram."
        ),
    ),
    OnboardingStep(
        id="mitre_threats", page=TAB_MITRE, is_page_intro=True,
        title="MITRE & Threats -- the organization's whole threat map",
        story=(
            "Zooming out from one investigation to the whole organization: which attacker techniques are "
            "actually showing up here, and how often?"
        ),
        basic=(
            "An ATT&CK-Navigator-style matrix of every tactic and technique this system has actually "
            "observed, org-wide -- not a theoretical checklist, real counts from real flagged events."
        ),
        advanced=(
            "Blank cells mean that (tactic, technique) pair genuinely has zero events in the current filter "
            "-- not that it's unsupported. This dataset models 5 of the 10 MITRE Enterprise tactics, stated "
            "explicitly rather than implying broader coverage."
        ),
    ),
    OnboardingStep(
        id="model_drift", page=TAB_MODEL, is_page_intro=True,
        title="Model & Drift -- trusting the machine behind the curtain",
        story=(
            "Every good analyst eventually asks the meta-question: how much should I actually trust this "
            "AI? This page is the honest answer."
        ),
        basic=(
            "Real, measured model performance -- precision, recall, calibration (does an 80% confidence "
            "score actually mean 80% right?), and concept drift detection (has user behavior shifted "
            "enough that the model's assumptions are stale?)."
        ),
        advanced=(
            "Includes bootstrap 95% confidence intervals and paired significance testing between imbalance-"
            "handling methods -- so a claim like 'SMOTE is better' is backed by a p-value, not just a "
            "slightly higher point estimate. ADWIN drift detection is evaluated against a real ground-truth "
            "drift log, with an honest detection-lag number, not an assumed one.\n\n"
            "This is also where the honest paper trail lives for the system's hardest problem: "
            "`lateral_movement` and `device_spoofing` are rare by construction (a spoofing campaign IS one "
            "anomalous login), so rare that both scored near-zero recall in every earlier evaluation. "
            "Adding session-window graph features and re-tuning per-class thresholds raised "
            "`lateral_movement` recall from 0% to 50% (2 of 4 test examples) on `xgboost_smote`, and lifted "
            "`device_spoofing` precision from 1.6% to 21.4% (a 13x reduction in false positives at the same "
            "75% recall) as a side benefit -- reported with equal honesty, `class_weight`'s `device_spoofing` "
            "recall regressed from 25% to 0% under the same change, and `lateral_movement` still isn't fully "
            "solved, because sample scarcity is the fundamental limit here, not a modeling shortcoming. Full "
            "writeup: `docs/phase_5_recall_investigation.md`."
        ),
    ),
    OnboardingStep(
        id="risk_heatmap", page=TAB_HEATMAP, is_page_intro=True,
        title="Risk Heatmap -- where is the organization actually exposed?",
        story=(
            "Your final stop: stepping back from individuals entirely to ask, as a security leader would, "
            "'which parts of this organization carry the most risk right now?'"
        ),
        basic=(
            "Aggregated risk by department and severity -- helps prioritize where to focus defensive "
            "attention and resourcing, not just which single account to investigate next."
        ),
        advanced=(
            "Built from the same live-filtered, threshold-driven flagged-event set as the Overview tab -- "
            "moving the sidebar filters or threshold slider updates this heatmap too, so it's never stale "
            "relative to what you're actively investigating."
        ),
    ),
]

STEP_BY_ID: dict[str, OnboardingStep] = {s.id: s for s in TOUR_STEPS}
TOTAL_STEPS = len(TOUR_STEPS)


# ---------------------------------------------------------------- FAQ coach --

ONBOARDING_FAQ: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("shap",),
        "SHAP explains why the AI made this prediction -- which specific factors pushed the risk score up "
        "or down, and by how much. Find it in Investigate -> Explainability.",
        "SHAP (SHapley Additive exPlanations) estimates each feature's contribution to the model's output "
        "using an additive attribution scheme rooted in cooperative game theory -- the sum of all "
        "contributions plus a baseline equals the model's actual prediction, so the explanation is "
        "mathematically consistent with the real score, not a separate approximation dressed up as one.",
    ),
    (
        ("mitre", "att&ck", "attack framework"),
        "MITRE ATT&CK is a shared, public vocabulary security teams use to describe attacker behavior -- "
        "'tactics' (the attacker's goal, like Credential Access) and 'techniques' (how they achieve it). "
        "See it applied to real events in Investigate and org-wide in MITRE & Threats.",
        "This dataset maps 5 of the 10 MITRE Enterprise tactics (Credential Access, Initial Access, "
        "Privilege Escalation, Lateral Movement, Defense Evasion), sourced from a maintained mapping table "
        "(`configs/mitre_mapping.yaml`), not inferred per-event -- stated honestly rather than implying "
        "full-matrix coverage.",
    ),
    (
        ("concept drift", "drift"),
        "Concept drift is when normal user/device behavior genuinely changes over time (a policy change, "
        "a new office, a shift to remote work) -- a model trained on old 'normal' can start misreading the "
        "new normal as anomalous. See it monitored in Overview and Model & Drift.",
        "Detected here with ADWIN (Adaptive Windowing), a streaming statistical test that flags a "
        "significant shift in a monitored signal's distribution -- evaluated here against a real "
        "ground-truth drift log with an honest measured detection lag (currently 0.49 days for the "
        "'remote work shift' scenario), not an assumed instant-detection claim.",
    ),
    (
        ("class imbalance", "imbalance"),
        "Real attacks are rare compared to normal activity -- if untreated, a model can get 99%+ 'accuracy' "
        "by just calling everything benign. This system compares 3 different techniques for handling that "
        "(no correction, class weighting, and SMOTE oversampling) side by side in Model & Drift.",
        "SMOTE synthesizes new minority-class training examples by interpolating between real neighbors; "
        "class weighting instead reweights the training loss so rare-class mistakes cost more. Both are "
        "compared with real significance testing (paired bootstrap on PR-AUC), not just point estimates.",
    ),
    (
        ("cold start", "cold-start", "new user"),
        "A brand-new user has no history yet to compare against -- 'cold start.' This system handles that "
        "by substituting a department-level typical baseline until the user builds up their own history.",
        "See `feature_engineering/cold_start.py` -- users within a configurable window of their join date "
        "get department-prior imputation for cold-start-eligible features, rather than an artificially "
        "anomalous raw sentinel or an artificially normal zero.",
    ),
    (
        ("false positive", "false positives"),
        "A false positive is a benign event the model incorrectly flagged as an attack -- too many of "
        "these and analysts stop trusting alerts entirely ('alert fatigue'). The Overview and Model & Drift "
        "tabs report false-positives-per-day honestly, alongside recall, not recall alone.",
        "This system's own stress-testing (see the robustness analysis) found precision can degrade "
        "sharply under noisy real-world-style input -- a real, disclosed limitation, not glossed over.",
    ),
    (
        ("nist", "cis controls", "cis"),
        "In the recommendation engine (Investigate -> AI analyst recommendation), each suggested response "
        "action is tagged with the real, published NIST CSF and CIS Controls v8 category it maps to -- "
        "context for how a recommended action fits into standard security frameworks.",
        "These are illustrative mappings to real published category names/IDs (`configs/response_"
        "framework_mapping.yaml`), not a formal compliance attestation -- a real control owner would still "
        "confirm the exact mapping against their own implementation.",
    ),
]


def _match_faq(question: str) -> tuple[str, str] | None:
    q = question.lower()
    for keywords, basic, advanced in ONBOARDING_FAQ:
        if any(k in q for k in keywords):
            return basic, advanced
    return None


def _contextual_why_flagged(question: str, context: dict | None) -> str | None:
    """If the user is mid-investigation and asks 'why was this flagged,'
    answer with THIS event's real evidence (reusing app.py's own
    evidence-gathering, not a fabricated canned response).
    """
    q = question.lower()
    if "why" not in q or "flag" not in q:
        return None
    if not context or not context.get("evidence"):
        return (
            "Select a flagged event in the Investigate tab first, and I can tell you exactly which "
            "evidence signals contributed to it."
        )
    lines = [f"For the event you're currently viewing, {len(context['evidence'])} evidence signal(s) fired:"]
    for e in context["evidence"][:6]:
        lines.append(f"- {e['text']}")
    return "\n".join(lines)


def answer_question(question: str, context: dict | None = None) -> str:
    contextual = _contextual_why_flagged(question, context)
    if contextual is not None:
        return contextual
    match = _match_faq(question)
    if match is not None:
        basic, _advanced = match
        return basic
    return (
        "I don't have a canned answer for that specific question yet, but here's what I can help with: "
        "SHAP explainability, MITRE ATT&CK, concept drift, class imbalance, cold-start handling, false "
        "positives, and NIST/CIS control mappings -- or ask \"why was this flagged\" while investigating "
        "a specific event."
    )


# ---------------------------------------------------------------- state --

def _default_state() -> dict:
    return {
        "dont_show_again": False,
        "seen_intro": False,
        "tour_active": False,
        "completed": False,
        "step_index": 0,
        "completed_ids": [],
        "advanced_by_step": {},
    }


def _load_persisted() -> dict:
    if ONBOARDING_STATE_PATH.exists():
        try:
            return {**_default_state(), **json.loads(ONBOARDING_STATE_PATH.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, OSError):
            return _default_state()
    return _default_state()


def _persist(state: dict) -> None:
    # Deliberately does NOT persist "tour_active" -- a brand new browser
    # session (a fresh page load, not just a rerun within one) should never
    # silently teleport the user back into a mid-tour spotlight/tab-jump
    # they didn't ask for in THIS session. Landing on "walkthrough paused at
    # step N, Resume?" (see render_tour_panel) is the adaptive-resume
    # experience requirement #7/#9 actually asks for -- remember progress,
    # but let the user opt back in rather than forcing it on them.
    try:
        ONBOARDING_STATE_PATH.write_text(
            json.dumps({
                "dont_show_again": state["dont_show_again"],
                "seen_intro": state["seen_intro"],
                "completed": state["completed"],
                "step_index": state["step_index"],
                "completed_ids": state["completed_ids"],
            }),
            encoding="utf-8",
        )
    except OSError:
        pass  # persistence is a nice-to-have; never break the dashboard over it


def init_onboarding_state() -> None:
    if "onboarding" not in st.session_state:
        st.session_state["onboarding"] = _load_persisted()


def _state() -> dict:
    init_onboarding_state()
    return st.session_state["onboarding"]


# ---------------------------------------------------------------- navigation --

def _ensure_active_tab(target_label: str) -> None:
    """Idempotently makes sure the browser's currently-visible tab matches
    `target_label` -- called on EVERY render while the tour is active (not
    a one-shot flag fired only right after a Next/Back click). This is
    deliberately self-healing rather than fire-once: a one-shot approach
    (queue a switch, consume it on the very next rerun) turned out to be
    fragile in practice -- any rerun that doesn't cleanly consume the queued
    flag (e.g. one triggered from inside an `@st.dialog`'s own callback, or
    simply a click landing while a previous switch's retry loop was still
    in flight) leaves it to fire LATE, on some unrelated later interaction
    (like "Tell me more"), visibly yanking the user to whatever page was
    queued long before -- exactly the "Tell me more sends me to the home
    page" and "Back/Next don't stay in sync" reports this replaced.

    Checking `aria-selected` before clicking makes repeated calls on
    already-correct pages a true no-op, so this can safely run on every
    single render without flicker or fighting a user's own manual clicks
    once they've exited the tour (this function is simply never called
    when `tour_active` is false).
    """
    components.html(
        f"""
        <script>
        (function() {{
            function ensure(attempts) {{
                const doc = window.parent.document;
                const tabs = doc.querySelectorAll('[data-testid="stTab"]');
                for (const tab of tabs) {{
                    if (tab.innerText.trim() === {json.dumps(target_label)}) {{
                        if (tab.getAttribute('aria-selected') !== 'true') {{
                            tab.click();
                        }}
                        return;
                    }}
                }}
                if (attempts > 0) setTimeout(function() {{ ensure(attempts - 1); }}, 150);
            }}
            ensure(20);
        }})();
        </script>
        """,
        height=0,
    )


def _go_to_step(new_index: int) -> None:
    state = _state()
    state["step_index"] = new_index
    if new_index >= TOTAL_STEPS:
        state["tour_active"] = False
        state["completed"] = True
    _persist(state)


# ---------------------------------------------------------------- dialogs --

@st.dialog("Welcome to the Identity Threat Detection Platform")
def _welcome_dialog() -> None:
    st.write(
        "I'll guide you through the dashboard step by step. By the end of this walkthrough you'll "
        "understand how the AI detects cyber threats, explains its decisions, and helps analysts "
        "investigate incidents."
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Start Guided Tour", type="primary", width="stretch"):
        state = _state()
        state["seen_intro"] = True
        state["tour_active"] = True
        _go_to_step(0)
        st.rerun()
    if c2.button("Skip", width="stretch"):
        state = _state()
        state["seen_intro"] = True
        _persist(state)
        st.rerun()
    if c3.button("Don't show again", width="stretch"):
        state = _state()
        state["seen_intro"] = True
        state["dont_show_again"] = True
        _persist(state)
        st.rerun()


@st.dialog("Congratulations!")
def _completion_dialog() -> None:
    st.write("You have completed the analyst onboarding. You now know how to:")
    st.markdown(
        "- Monitor live threats\n"
        "- Investigate incidents\n"
        "- Understand AI explanations\n"
        "- Interpret behavioral anomalies\n"
        "- Use MITRE ATT&CK mappings\n"
        "- Monitor model health\n"
        "- Assess organizational cyber risk"
    )
    c1, c2 = st.columns(2)
    if c1.button("Start Exploring", type="primary", width="stretch"):
        st.session_state["onboarding_show_completion"] = False
        st.rerun()
    if c2.button("Restart Tour", width="stretch"):
        state = _state()
        state["tour_active"] = True
        state["completed"] = False
        st.session_state["onboarding_show_completion"] = False
        _go_to_step(0)
        st.rerun()


def maybe_show_welcome() -> None:
    state = _state()
    if not state["seen_intro"] and not state["dont_show_again"]:
        _welcome_dialog()
    if st.session_state.pop("onboarding_show_completion", False):
        _completion_dialog()


# ---------------------------------------------------------------- sidebar tour panel --

def render_tour_panel() -> None:
    """The persistent mentor/narrator panel -- lives in the sidebar so it's
    visible regardless of which tab is currently open (st.tabs gives
    Python no visibility into the selected tab, so a sidebar-anchored
    control is the one placement guaranteed stable across tabs).

    Deliberately NEVER disappears entirely, even after completion or
    "Don't show again" -- that setting only suppresses the AUTOMATIC
    welcome popup on load, not the analyst's ability to reopen the guide
    whenever they actually want it later.
    """
    state = _state()

    if not state["tour_active"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Guide")
        if state["completed"]:
            st.sidebar.caption("You've completed the guided walkthrough.")
            if st.sidebar.button("Restart Tour", width="stretch"):
                state["completed"] = False
                state["tour_active"] = True
                _go_to_step(0)
                st.rerun()
        elif state["seen_intro"] and state["step_index"] > 0:
            st.sidebar.caption(f"Walkthrough paused at step {state['step_index'] + 1} of {TOTAL_STEPS}.")
            rc1, rc2 = st.sidebar.columns(2)
            if rc1.button("Resume", type="primary", width="stretch"):
                state["tour_active"] = True
                _persist(state)
                st.rerun()
            if rc2.button("Start over", width="stretch"):
                state["tour_active"] = True
                _go_to_step(0)
                st.rerun()
        else:
            st.sidebar.caption("New here? Take a guided walkthrough of the dashboard.")
            if st.sidebar.button("Start Guided Tour", type="primary", width="stretch"):
                state["seen_intro"] = True
                state["tour_active"] = True
                _go_to_step(0)
                st.rerun()
        return

    idx = state["step_index"]
    if idx >= TOTAL_STEPS:
        st.session_state["onboarding_show_completion"] = True
        st.rerun()
        return

    step = TOUR_STEPS[idx]
    _ensure_active_tab(step.page)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Guided Walkthrough")
    st.sidebar.progress((idx + 1) / TOTAL_STEPS, text=f"Step {idx + 1} of {TOTAL_STEPS}")
    with st.sidebar.container(border=True):
        st.markdown(f"**{step.title}**")
        st.caption(step.story)
        st.write(step.basic)
        advanced_map = state["advanced_by_step"]
        show_advanced = advanced_map.get(step.id, False)
        if st.button("Tell me more" if not show_advanced else "Show less", key=f"ob_adv_{step.id}"):
            advanced_map[step.id] = not show_advanced
            _persist(state)
            st.rerun()
        if show_advanced:
            st.info(step.advanced)

    nav1, nav2, nav3 = st.sidebar.columns(3)
    if nav1.button("Back", disabled=idx == 0, width="stretch"):
        state["completed_ids"] = list(set(state["completed_ids"]) - {step.id})
        _go_to_step(max(0, idx - 1))
        st.rerun()
    if nav2.button("Next", type="primary", width="stretch"):
        if step.id not in state["completed_ids"]:
            state["completed_ids"].append(step.id)
        _go_to_step(idx + 1)
        st.rerun()
    if nav3.button("Exit", width="stretch"):
        state["tour_active"] = False
        _persist(state)
        st.rerun()


# ---------------------------------------------------------------- spotlight/dim --

def spotlight(step_id: str, render_fn: Callable[[], None]) -> None:
    """Wraps one section of a page: if the guided tour is active and its
    CURRENT step is `step_id`, render with a glowing highlight; if the
    tour is active on this SAME page but focused on a different step,
    render dimmed (de-emphasized, not hidden); otherwise (tour inactive,
    or the tour is currently on a different page entirely) render exactly
    as it would without onboarding -- this feature must never change the
    app's behavior when it isn't actively teaching.
    """
    state = _state()
    step = STEP_BY_ID[step_id]
    active = state["tour_active"] and 0 <= state["step_index"] < TOTAL_STEPS
    current = TOUR_STEPS[state["step_index"]] if active else None

    if not active or current is None or current.page != step.page:
        render_fn()
        return

    if current.id == step_id:
        with st.container(key=f"ob_spot_{step_id}", border=True):
            render_fn()
    else:
        with st.container(key=f"ob_dim_{step_id}"):
            render_fn()


def page_intro_wrap(step_id: str, render_fn: Callable[[], None]) -> None:
    """Same idea as `spotlight`, for pages with exactly ONE tour step
    (the whole tab body is the highlight -- no per-widget dimming needed
    since there's nothing else on that page to de-emphasize)."""
    state = _state()
    step = STEP_BY_ID[step_id]
    active = state["tour_active"] and 0 <= state["step_index"] < TOTAL_STEPS
    current = TOUR_STEPS[state["step_index"]] if active else None

    if active and current is not None and current.id == step_id:
        with st.container(key=f"ob_spot_{step_id}", border=True):
            st.markdown(f"#### 🧭 {step.title}")
            render_fn()
    else:
        render_fn()


# ---------------------------------------------------------------- floating coach --

def render_coach(context: dict | None = None, key_suffix: str = "global") -> None:
    with st.popover("💬 Need Help?", width="stretch"):
        st.caption("Ask me anything about this dashboard -- e.g. \"What is SHAP?\", \"What is MITRE?\", \"Why was this flagged?\"")
        question = st.text_input(
            "Your question", key=f"ob_coach_question_{key_suffix}",
            label_visibility="collapsed", placeholder="Ask a question...",
        )
        if question:
            st.markdown(answer_question(question, context))
        st.caption("Answers are generated from this dashboard's own real content -- not a live external AI service.")
