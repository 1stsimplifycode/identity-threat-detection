"""Shared visual identity for the analyst dashboard -- an enterprise/
industrial security-ops palette: neutral ground, blue as the one accent,
red/amber/green reserved specifically for severity (no other use of those
hues anywhere in the UI). No gradients, no decorative illustration, no
emoji as UI markers -- status is always a text+color chip, never a symbol
alone (color-blind-safe by design).

Supports both a dark and a light appearance (`PALETTES["dark"]` /
`PALETTES["light"]`), toggled at runtime via `st.session_state["theme"]` --
NOT via `.streamlit/config.toml` (that file is Streamlit's own base widget
theme and stays untouched; instead, exactly like the existing precedent of
overriding `[data-testid="stAppViewContainer"]`/`[data-testid="stSidebar"]`
below, every themed surface is repainted with custom CSS keyed off the
selected palette). Callers that build Plotly figures must pull colors from
the SAME palette dict (`get_palette(theme)`) rather than any hardcoded hex,
or a chart would stay dark-tuned (invisible text/gridlines) after switching
to the light appearance -- see `dashboard/app.py`'s render functions, all
of which now take a `palette` argument for exactly this reason.

`.streamlit/config.toml` sets the base Streamlit widget palette (buttons,
sliders, inputs) for the DARK appearance specifically; native widget chrome
is best-effort themed for light mode via the CSS overrides below, since
Streamlit doesn't expose a supported way to swap its base theme at runtime
without touching that config file.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------- palettes --

PALETTES: dict[str, dict] = {
    "dark": {
        "bg_base": "#0B0E14",
        "bg_panel": "#141920",
        "bg_panel_alt": "#1B222C",
        "border": "#262D38",
        "text_primary": "#E6EAF0",
        "text_secondary": "#8B95A5",
        "text_muted": "#5B6472",
        "accent": "#3B82F6",
        "accent_soft": "#16233A",
        "neutral_chip": "#2B3341",
        "neutral_chip_text": "#8B95A5",
        # Colorblind-safe severity scale: "low" is deliberately teal/blue, not
        # green -- red-green is by far the most common form of color vision
        # deficiency (deuteranopia/protanopia, ~8% of men), and a red-vs-green
        # "critical vs. low" pair is close to indistinguishable for those
        # users. Every severity/status indicator that uses this palette also
        # carries a text label (CRITICAL/HIGH/MEDIUM/LOW, "detected", etc.)
        # so color is reinforcing, never the only signal -- but the hues
        # themselves are chosen so an at-a-glance scan still works without
        # relying on red/green discrimination at all.
        "severity_colors": {
            "critical": "#E5484D", "high": "#F2934D", "medium": "#E8C339", "low": "#2DB4C4",
        },
        "severity_soft": {
            "critical": "#3A1A1C", "high": "#3A2716", "medium": "#38300F", "low": "#123338",
        },
    },
    "light": {
        "bg_base": "#F4F6F9",
        "bg_panel": "#FFFFFF",
        "bg_panel_alt": "#EFF2F6",
        "border": "#D7DDE6",
        "text_primary": "#161B22",
        "text_secondary": "#4B5566",
        "text_muted": "#7C8598",
        "accent": "#2054C9",
        "accent_soft": "#DCE6FA",
        "neutral_chip": "#E7EBF1",
        "neutral_chip_text": "#4B5566",
        "severity_colors": {
            "critical": "#C22A32", "high": "#B0591A", "medium": "#8A6A06", "low": "#0D7A8A",
        },
        "severity_soft": {
            "critical": "#FBE1E2", "high": "#FAE6D6", "medium": "#FAF0CB", "low": "#DAF0F3",
        },
    },
}


def get_palette(theme: str) -> dict:
    return PALETTES.get(theme, PALETTES["dark"])


def current_theme() -> str:
    return st.session_state.get("itd_theme", "dark")


def theme_toggle() -> str:
    """Sidebar dark/light toggle. Returns the active theme name; also
    stores it to session_state so every render function in the same run
    can call `current_theme()` independently.
    """
    theme = st.sidebar.radio(
        "Appearance", options=["dark", "light"],
        horizontal=True, key="itd_theme", help="Switches the dashboard's color scheme.",
    )
    return theme


FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
"""


def _build_css(p: dict) -> str:
    return f"""
<style>
{FONT_CSS}

html, body, [class*="css"] {{
    font-family: "IBM Plex Sans", -apple-system, "Segoe UI", sans-serif;
}}

.itd-mono {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
}}

/* -- App chrome -- */
[data-testid="stAppViewContainer"] {{
    background-color: {p["bg_base"]};
}}
[data-testid="stHeader"] {{
    background-color: {p["bg_base"]};
}}
[data-testid="stSidebar"] {{
    background-color: {p["bg_panel"]};
    border-right: 1px solid {p["border"]};
}}
[data-testid="stSidebar"] * {{
    color: {p["text_primary"]};
}}
h1, h2, h3, p, span, label, .stMarkdown {{
    color: {p["text_primary"]};
}}
h1, h2, h3 {{
    font-weight: 600;
    letter-spacing: -0.01em;
}}
h1 {{ font-size: 1.5rem; }}
h2 {{ font-size: 1.15rem; color: {p["text_primary"]}; }}
h3 {{ font-size: 0.95rem; color: {p["text_secondary"]}; text-transform: uppercase; letter-spacing: 0.06em; }}

/* -- Native widget chrome (best-effort light/dark repaint; base widget
   theme itself stays whatever .streamlit/config.toml sets) -- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background-color: {p["bg_panel"]};
    border: 1px solid {p["border"]};
    border-radius: 6px;
}}
.stButton > button, .stDownloadButton > button {{
    background-color: {p["bg_panel_alt"]};
    color: {p["text_primary"]};
    border: 1px solid {p["border"]};
}}
[data-testid="stMetricValue"] {{ color: {p["text_primary"]}; }}
[data-testid="stMetricLabel"] {{ color: {p["text_secondary"]}; }}
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    border-bottom: 1px solid {p["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    color: {p["text_secondary"]};
    font-size: 0.85rem;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    color: {p["accent"]};
}}

/* -- Form controls (multiselect/selectbox/text input/slider) -- BaseWeb
   renders these with its own dark-tuned inline styling that plain
   inherited color doesn't override, so these need !important to actually
   repaint under the light palette instead of leaving black-on-black text. -- */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {{
    background-color: {p["bg_panel_alt"]} !important;
    border-color: {p["border"]} !important;
}}
[data-baseweb="select"] div, [data-baseweb="select"] span,
[data-baseweb="input"] input, [data-baseweb="base-input"] input {{
    color: {p["text_primary"]} !important;
}}
[data-baseweb="tag"] {{
    background-color: {p["accent_soft"]} !important;
}}
[data-baseweb="tag"] span {{
    color: {p["text_primary"]} !important;
}}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{
    color: {p["text_secondary"]} !important;
}}
/* Text input (react-aria-components, not BaseWeb, in current Streamlit --
   `.react-aria-TextField` is a stable framework class name, unlike the
   emotion-hashed class Streamlit puts on the same element, so it's the
   only reliable selector here). */
[data-testid="stTextInput"] .react-aria-TextField > div, [data-testid="stTextInput"] input {{
    background-color: {p["bg_panel_alt"]} !important;
    color: {p["text_primary"]} !important;
    border-color: {p["border"]} !important;
}}
/* Inline `code` pills (e.g. the sidebar's `small_dev` / `xgboost_smote`
   run caption) -- Streamlit's default inline-code background is tuned
   for the dark base theme and doesn't repaint with the rest of the page,
   which left dark text on a near-black pill (unreadable) under light. */
code {{
    background-color: {p["bg_panel_alt"]} !important;
    color: {p["text_primary"]} !important;
}}

/* -- Disclaimer banner -- */
.itd-disclaimer {{
    background-color: {p["accent_soft"]};
    border: 1px solid {p["accent"]};
    border-radius: 4px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    color: {p["text_secondary"]};
    margin-bottom: 1rem;
}}
.itd-disclaimer b {{ color: {p["text_primary"]}; }}

/* -- KPI cards -- */
.itd-kpi-row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.itd-kpi-card {{
    background-color: {p["bg_panel"]};
    border: 1px solid {p["border"]};
    border-radius: 6px;
    padding: 0.75rem 1rem;
    min-width: 150px;
    flex: 1 1 150px;
}}
.itd-kpi-label {{
    font-size: 0.72rem;
    color: {p["text_secondary"]};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
}}
.itd-kpi-value {{
    font-family: "IBM Plex Mono", monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: {p["text_primary"]};
    font-variant-numeric: tabular-nums;
}}
.itd-kpi-sub {{ font-size: 0.72rem; color: {p["text_muted"]}; margin-top: 0.15rem; }}

/* -- Section-scope labels: distinguishes a run-level (fixed) metric group
   from a current-filter (live) metric group, so the two are never
   mistaken for each other at a glance. -- */
.itd-scope-label {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: {p["text_muted"]}; margin-bottom: 0.4rem;
}}
.itd-scope-label.live {{ color: {p["accent"]}; }}
.itd-scope-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; display: inline-block; }}

/* -- Status / severity chips -- */
.itd-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
}}
.itd-chip-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}

/* -- Panels / section containers -- */
.itd-panel {{
    background-color: {p["bg_panel"]};
    border: 1px solid {p["border"]};
    border-radius: 6px;
    padding: 1rem;
}}

/* -- Explanation blocks -- */
.itd-explanation {{
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem;
    background-color: {p["bg_panel_alt"]};
    border-left: 3px solid {p["accent"]};
    padding: 0.6rem 0.8rem;
    border-radius: 3px;
    color: {p["text_primary"]};
    line-height: 1.5;
}}
.itd-explanation-approx {{
    border-left: 3px solid {p["text_muted"]};
    color: {p["text_secondary"]};
}}

/* -- Rationale callout (attack campaign's plain-English explanation) -- */
.itd-rationale {{
    background-color: {p["accent_soft"]};
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    color: {p["text_primary"]};
    font-style: italic;
}}

hr {{ border-color: {p["border"]}; }}

/* -- Section dividers: a labeled rule marking a major hand-off in the
   page's narrative (e.g. "operational posture" -> "model metrics"),
   stronger than a bare <hr> so the reader feels the section change. -- */
.itd-section-divider {{
    display: flex; align-items: center; gap: 0.75rem;
    margin: 1.75rem 0 1rem 0;
}}
.itd-section-divider::after {{
    content: ""; flex: 1; height: 1px; background: {p["border"]};
}}
.itd-section-divider-label {{
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: {p["text_secondary"]}; white-space: nowrap;
}}

/* -- Hero KPI cards: the operational-posture row on Overview -- larger,
   bolder, with a tone-colored left rail so severity reads at a glance
   even before the number is parsed. -- */
.itd-hero-row {{ display: flex; gap: 0.85rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
.itd-hero-card {{
    background-color: {p["bg_panel"]};
    border: 1px solid {p["border"]};
    border-left: 4px solid var(--itd-tone, {p["accent"]});
    border-radius: 8px;
    padding: 1rem 1.15rem;
    min-width: 175px;
    flex: 1 1 175px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}}
.itd-hero-label {{
    font-size: 0.72rem; color: {p["text_secondary"]}; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 0.3rem; font-weight: 600;
}}
.itd-hero-value {{
    font-family: "IBM Plex Mono", monospace; font-size: 1.9rem; font-weight: 700;
    color: {p["text_primary"]}; font-variant-numeric: tabular-nums; line-height: 1.1;
}}
.itd-hero-sub {{ font-size: 0.76rem; color: {p["text_muted"]}; margin-top: 0.3rem; }}

/* -- Investigation workflow step cards -- */
.itd-step-header {{
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem;
}}
.itd-step-badge {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.6rem; height: 1.6rem; border-radius: 50%;
    background: {p["accent_soft"]}; color: {p["accent"]};
    font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
}}
.itd-step-title {{ font-size: 1.02rem; font-weight: 700; color: {p["text_primary"]}; }}
.itd-step-subtitle {{ font-size: 0.78rem; color: {p["text_secondary"]}; margin-left: 2.2rem; margin-top: -0.4rem; margin-bottom: 0.5rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 10px !important;
}}

/* -- AI investigation narrative -- */
.itd-narrative {{
    font-size: 0.98rem; line-height: 1.65; color: {p["text_primary"]};
    background: {p["accent_soft"]}; border-left: 3px solid {p["accent"]};
    border-radius: 4px; padding: 0.9rem 1.1rem; margin-bottom: 0.75rem;
}}

/* -- Risk score gauge-style readout -- */
.itd-risk-readout {{
    display: flex; align-items: baseline; gap: 0.5rem;
}}
.itd-risk-value {{
    font-family: "IBM Plex Mono", monospace; font-weight: 700; font-size: 2.1rem;
    color: var(--itd-tone, {p["text_primary"]});
}}
.itd-risk-unit {{ font-size: 0.82rem; color: {p["text_secondary"]}; }}

/* -- Explainability comparison table -- */
.itd-explain-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
.itd-explain-table th {{
    text-align: left; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {p["text_secondary"]}; padding: 0.4rem 0.6rem; border-bottom: 1px solid {p["border"]};
}}
.itd-explain-table td {{
    padding: 0.45rem 0.6rem; border-bottom: 1px solid {p["border"]};
    color: {p["text_primary"]}; font-family: "IBM Plex Mono", monospace; font-size: 0.82rem;
}}
.itd-explain-table td:first-child {{ font-family: "IBM Plex Sans", sans-serif; font-weight: 600; }}

/* -- Attack coverage strip -- */
.itd-coverage-row {{ display: flex; gap: 0.7rem; flex-wrap: wrap; }}
.itd-coverage-card {{
    background: {p["bg_panel"]}; border: 1px solid {p["border"]}; border-radius: 8px;
    padding: 0.75rem 0.9rem; flex: 1 1 170px; min-width: 170px;
}}
.itd-coverage-card.covered {{ border-color: {p["accent"]}; }}
.itd-coverage-name {{ font-size: 0.8rem; font-weight: 700; color: {p["text_primary"]}; margin-bottom: 0.35rem; }}
.itd-coverage-meta {{ font-size: 0.72rem; color: {p["text_secondary"]}; }}

/* -- Threat-level banner: the single "how bad is it right now" readout
   atop the Executive Overview. -- */
@keyframes itd-fade-up {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .itd-fade-up, .itd-timeline-node, .itd-pipeline-stage {{ animation: none !important; }}
}}
.itd-threat-banner {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
    background: {p["bg_panel"]}; border: 1px solid {p["border"]};
    border-left: 6px solid var(--itd-tone, {p["accent"]});
    border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}}
.itd-threat-level {{
    font-size: 1.6rem; font-weight: 800; color: var(--itd-tone, {p["text_primary"]});
    letter-spacing: -0.01em;
}}
.itd-threat-sub {{ font-size: 0.8rem; color: {p["text_secondary"]}; margin-top: 0.15rem; }}
.itd-threat-meta-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
.itd-threat-meta {{ text-align: right; }}
.itd-threat-meta-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: {p["text_secondary"]}; }}
.itd-threat-meta-value {{ font-family: "IBM Plex Mono", monospace; font-weight: 700; font-size: 1.15rem; color: {p["text_primary"]}; }}

/* -- Vertical incident timeline -- */
.itd-timeline {{ position: relative; padding-left: 1.6rem; }}
.itd-timeline::before {{
    content: ""; position: absolute; left: 0.4rem; top: 0.3rem; bottom: 0.3rem; width: 2px;
    background: {p["border"]};
}}
.itd-timeline-node {{
    position: relative; margin-bottom: 1.1rem; opacity: 0; animation: itd-fade-up 0.4s ease-out forwards;
}}
.itd-timeline-node::before {{
    content: ""; position: absolute; left: -1.6rem; top: 0.2rem; width: 10px; height: 10px; border-radius: 50%;
    background: var(--itd-tone, {p["accent"]}); border: 2px solid {p["bg_base"]}; box-shadow: 0 0 0 2px var(--itd-tone, {p["accent"]});
}}
.itd-timeline-time {{ font-size: 0.68rem; color: {p["text_muted"]}; font-family: "IBM Plex Mono", monospace; }}
.itd-timeline-label {{ font-size: 0.9rem; font-weight: 700; color: {p["text_primary"]}; margin: 0.1rem 0; }}
.itd-timeline-detail {{ font-size: 0.78rem; color: {p["text_secondary"]}; }}

/* -- Risk driver cards (positive / negative SHAP contributors) -- */
.itd-driver-row {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.itd-driver-col {{ flex: 1 1 260px; min-width: 240px; }}
.itd-driver-col-title {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }}
.itd-driver-card {{
    display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
    background: {p["bg_panel_alt"]}; border-left: 3px solid var(--itd-tone, {p["accent"]});
    border-radius: 4px; padding: 0.5rem 0.7rem; margin-bottom: 0.4rem;
}}
.itd-driver-name {{ font-size: 0.82rem; color: {p["text_primary"]}; }}
.itd-driver-value {{ font-family: "IBM Plex Mono", monospace; font-weight: 700; font-size: 0.82rem; color: var(--itd-tone, {p["text_primary"]}); white-space: nowrap; }}

/* -- Recommendation action cards -- */
.itd-rec-row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; }}
.itd-rec-card {{
    flex: 1 1 240px; min-width: 220px; background: {p["bg_panel"]}; border: 1px solid {p["border"]};
    border-top: 3px solid var(--itd-tone, {p["accent"]}); border-radius: 8px; padding: 0.85rem 1rem;
}}
.itd-rec-action {{ font-size: 0.88rem; font-weight: 700; color: {p["text_primary"]}; margin-bottom: 0.45rem; }}
.itd-rec-badges {{ display: flex; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
.itd-rec-badge {{
    font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    padding: 0.14rem 0.5rem; border-radius: 999px; background: {p["bg_panel_alt"]}; color: {p["text_secondary"]};
}}
.itd-rec-badge.tone {{ background: var(--itd-tone-soft, {p["accent_soft"]}); color: var(--itd-tone, {p["accent"]}); }}
.itd-rec-meta {{ font-size: 0.72rem; color: {p["text_secondary"]}; margin-top: 0.15rem; line-height: 1.4; }}
.itd-rec-why {{ font-size: 0.82rem; color: {p["text_primary"]}; margin: 0.3rem 0 0.7rem 0; }}
.itd-rec-why-item {{ font-size: 0.8rem; color: {p["text_secondary"]}; padding-left: 0.2rem; }}
.itd-rec-why-item.tick {{ color: {p["severity_colors"]["low"]}; }}

/* -- Detection pipeline diagram -- */
.itd-pipeline {{ display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; margin: 0.5rem 0; }}
.itd-pipeline-stage {{
    background: {p["bg_panel_alt"]}; border: 1px solid {p["border"]}; border-radius: 6px;
    padding: 0.55rem 0.8rem; font-size: 0.76rem; font-weight: 600; color: {p["text_secondary"]};
    flex: 1 1 auto; text-align: center; transition: all 0.3s ease;
}}
.itd-pipeline-stage.active {{
    border-color: {p["accent"]}; color: {p["accent"]}; background: {p["accent_soft"]};
    box-shadow: 0 0 0 3px {p["accent_soft"]};
}}
.itd-pipeline-arrow {{ color: {p["text_muted"]}; font-size: 0.9rem; flex: 0 0 auto; }}

/* -- Sticky side panel (User Risk Profile) -- */
.itd-sticky {{ position: sticky; top: 1rem; }}

/* -- Campaign cards -- */
.itd-campaign-card {{
    background: {p["bg_panel"]}; border: 1px solid {p["border"]}; border-left: 4px solid var(--itd-tone, {p["accent"]});
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
}}

/* -- Larger, more forgiving click target for the Appearance radio -- */
[data-testid="stRadio"] label {{ cursor: pointer; padding: 0.15rem 0.4rem; }}

/* -- Interactive onboarding guide (dashboard/onboarding.py) -- */
@keyframes itd-onboarding-pulse {{
    0%   {{ box-shadow: 0 0 0 0 {p["accent_soft"]}; }}
    70%  {{ box-shadow: 0 0 0 10px rgba(0,0,0,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
}}
[class*="st-key-ob_spot_"] {{
    border: 2px solid {p["accent"]} !important;
    border-radius: 10px; padding: 0.4rem 0.6rem 0.6rem;
    background: {p["accent_soft"]};
    animation: itd-onboarding-pulse 2s ease-out infinite;
    transition: all 0.4s ease;
}}
[class*="st-key-ob_dim_"] {{
    opacity: 0.38; filter: grayscale(35%);
    pointer-events: none;
    transition: opacity 0.4s ease, filter 0.4s ease;
}}
@media (prefers-reduced-motion: reduce) {{
    [class*="st-key-ob_spot_"] {{ animation: none; }}
}}

/* -- Responsive: tablet and narrower -- */
@media (max-width: 900px) {{
    .itd-kpi-card, .itd-hero-card {{ min-width: 130px; padding: 0.6rem 0.8rem; }}
    .itd-kpi-value {{ font-size: 1.2rem; }}
    .itd-hero-value {{ font-size: 1.5rem; }}
    h1 {{ font-size: 1.2rem; }}
    .itd-panel {{ padding: 0.7rem; }}
    .itd-step-subtitle {{ margin-left: 0; }}
    .itd-threat-banner {{ flex-direction: column; align-items: flex-start; }}
    .itd-sticky {{ position: static; }}
}}
</style>
"""


def inject_css(theme: str | None = None) -> None:
    p = get_palette(theme or current_theme())
    st.markdown(_build_css(p), unsafe_allow_html=True)


def severity_chip(severity: str | None, palette: dict) -> str:
    neutral, neutral_text, muted = palette["neutral_chip"], palette["neutral_chip_text"], palette["text_muted"]
    if not severity or (isinstance(severity, float)):
        return (
            f'<span class="itd-chip" style="background:{neutral}; color:{neutral_text};">'
            f'<span class="itd-chip-dot" style="background:{muted};"></span>N/A</span>'
        )
    sev = str(severity).lower()
    color = palette["severity_colors"].get(sev, palette["text_muted"])
    soft = palette["severity_soft"].get(sev, neutral)
    return (
        f'<span class="itd-chip" style="background:{soft}; color:{color};">'
        f'<span class="itd-chip-dot" style="background:{color};"></span>{sev.upper()}</span>'
    )


def status_chip(label: str, kind: str, palette: dict) -> str:
    """`kind`: "positive" (green), "negative" (red), "neutral" (gray) --
    for non-severity status (e.g. "detected" / "missed" / "benign").
    """
    sev_colors, sev_soft = palette["severity_colors"], palette["severity_soft"]
    lookup = {
        "positive": (sev_colors["low"], sev_soft["low"]),
        "negative": (sev_colors["critical"], sev_soft["critical"]),
        "neutral": (palette["neutral_chip_text"], palette["neutral_chip"]),
    }
    color, soft = lookup.get(kind, lookup["neutral"])
    return (
        f'<span class="itd-chip" style="background:{soft}; color:{color};">'
        f'<span class="itd-chip-dot" style="background:{color};"></span>{label}</span>'
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="itd-kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="itd-kpi-card"><div class="itd-kpi-label">{label}</div>'
        f'<div class="itd-kpi-value">{value}</div>{sub_html}</div>'
    )


def kpi_row(cards_html: list[str]) -> str:
    return f'<div class="itd-kpi-row">{"".join(cards_html)}</div>'


def section_divider(label: str) -> str:
    """Marks a hand-off between major narrative sections (e.g.
    "operational posture" -> "model metrics") -- stronger than a bare
    <hr> so the reader feels the section boundary, not just a rule.
    """
    return f'<div class="itd-section-divider"><span class="itd-section-divider-label">{label}</span></div>'


def hero_kpi_card(label: str, value: str, sub: str = "", tone: str | None = None) -> str:
    """A larger, bolder KPI card for the operational-posture row -- `tone`
    is a hex color (e.g. a severity color) rendered as a left rail so
    urgency reads before the number is even parsed; omit for neutral.
    """
    style = f' style="--itd-tone:{tone};"' if tone else ""
    sub_html = f'<div class="itd-hero-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="itd-hero-card"{style}><div class="itd-hero-label">{label}</div>'
        f'<div class="itd-hero-value">{value}</div>{sub_html}</div>'
    )


def hero_row(cards_html: list[str]) -> str:
    return f'<div class="itd-hero-row">{"".join(cards_html)}</div>'


def step_header(step_num: int, title: str, subtitle: str = "") -> str:
    """Header for one card in the Investigate tab's narrative workflow
    (Alert -> Event Details -> AI Summary -> ... -> MITRE Mapping) --
    a numbered badge so the sequence itself is visible, not just implied
    by vertical order.
    """
    sub_html = f'<div class="itd-step-subtitle">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="itd-step-header"><span class="itd-step-badge">{step_num}</span>'
        f'<span class="itd-step-title">{title}</span></div>{sub_html}'
    )


def narrative_block(text: str) -> str:
    return f'<div class="itd-narrative">{text}</div>'


def risk_readout(value: str, unit: str, tone: str | None = None) -> str:
    style = f' style="--itd-tone:{tone};"' if tone else ""
    return f'<div class="itd-risk-readout"{style}><span class="itd-risk-value">{value}</span><span class="itd-risk-unit">{unit}</span></div>'


def explain_table(rows: list[dict]) -> str:
    """`rows`: list of {"feature": str, "normal": str, "observed": str,
    "contribution": str} -- the three-way comparison item 5 of the brief
    asks for (normal vs. observed vs. contribution to the anomaly score),
    as one table instead of two separately-plotted charts.
    """
    body = "".join(
        f'<tr><td>{r["feature"]}</td><td>{r["normal"]}</td><td>{r["observed"]}</td><td>{r["contribution"]}</td></tr>'
        for r in rows
    )
    return (
        '<table class="itd-explain-table"><thead><tr>'
        "<th>Feature</th><th>Normal (baseline)</th><th>Observed (this event)</th><th>Contribution to risk score</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def coverage_card(name: str, covered: bool, count: str, sub: str) -> str:
    cls = "itd-coverage-card covered" if covered else "itd-coverage-card"
    return f'<div class="{cls}"><div class="itd-coverage-name">{name}</div><div class="itd-coverage-meta">{count}</div><div class="itd-coverage-meta">{sub}</div></div>'


def coverage_row(cards_html: list[str]) -> str:
    return f'<div class="itd-coverage-row">{"".join(cards_html)}</div>'


def threat_banner(level: str, sub: str, metas: list[tuple[str, str]], tone: str | None = None) -> str:
    """The single "how bad is it right now" readout atop the Executive
    Overview. `metas` is a list of (label, value) pairs shown to the right.
    """
    style = f' style="--itd-tone:{tone};"' if tone else ""
    metas_html = "".join(
        f'<div class="itd-threat-meta"><div class="itd-threat-meta-label">{label}</div><div class="itd-threat-meta-value">{value}</div></div>'
        for label, value in metas
    )
    return (
        f'<div class="itd-threat-banner"{style}>'
        f'<div><div class="itd-threat-level">{level}</div><div class="itd-threat-sub">{sub}</div></div>'
        f'<div class="itd-threat-meta-row">{metas_html}</div></div>'
    )


def timeline(nodes: list[dict]) -> str:
    """`nodes`: list of {"time": str, "label": str, "detail": str, "tone": hex|None}
    -- rendered as an animated vertical timeline, each node fading up with
    a staggered delay so the sequence itself reads as a reconstruction,
    not a static list.
    """
    items = []
    for i, n in enumerate(nodes):
        style = f'style="animation-delay:{i * 0.12:.2f}s; --itd-tone:{n.get("tone") or ""};"'
        items.append(
            f'<div class="itd-timeline-node" {style}>'
            f'<div class="itd-timeline-time">{n["time"]}</div>'
            f'<div class="itd-timeline-label">{n["label"]}</div>'
            f'<div class="itd-timeline-detail">{n.get("detail", "")}</div></div>'
        )
    return f'<div class="itd-timeline">{"".join(items)}</div>'


def driver_cards(items: list[tuple[str, str]], tone: str) -> str:
    """`items`: list of (name, formatted_value) -- one column of the
    positive/negative risk-driver display.
    """
    if not items:
        return '<div class="itd-driver-detail" style="opacity:0.6;">None</div>'
    cards = "".join(
        f'<div class="itd-driver-card" style="--itd-tone:{tone};">'
        f'<span class="itd-driver-name">{name}</span><span class="itd-driver-value">{value}</span></div>'
        for name, value in items
    )
    return cards


def driver_columns(positive_html: str, negative_html: str, positive_tone: str, negative_tone: str) -> str:
    return (
        '<div class="itd-driver-row">'
        f'<div class="itd-driver-col"><div class="itd-driver-col-title" style="color:{positive_tone};">Increases risk</div>{positive_html}</div>'
        f'<div class="itd-driver-col"><div class="itd-driver-col-title" style="color:{negative_tone};">Decreases risk</div>{negative_html}</div>'
        "</div>"
    )


def recommendation_cards(items: list[dict]) -> str:
    """`items`: list of {"action", "priority", "confidence", "evidence_count",
    "explanation", "tone", "tone_soft"} -- one card per evidence-corroborated
    candidate action, badges for priority/confidence/evidence-count so the
    strength of support is visible before reading the explanation text.
    Optional `"framework_line"`: a short string (e.g. published NIST CSF /
    CIS Controls category names this action maps to) rendered as a second,
    muted meta line under the explanation.
    """
    cards = []
    for r in items:
        style = f' style="--itd-tone:{r["tone"]}; --itd-tone-soft:{r.get("tone_soft", r["tone"])};"'
        badges = (
            f'<span class="itd-rec-badge tone">Priority: {r["priority"]}</span>'
            f'<span class="itd-rec-badge">Confidence: {r["confidence"]}</span>'
            f'<span class="itd-rec-badge">{r["evidence_count"]} evidence signal(s)</span>'
        )
        framework_html = f'<div class="itd-rec-meta">{r["framework_line"]}</div>' if r.get("framework_line") else ""
        cards.append(
            f'<div class="itd-rec-card"{style}><div class="itd-rec-action">{r["action"]}</div>'
            f'<div class="itd-rec-badges">{badges}</div>'
            f'<div class="itd-rec-meta">{r["explanation"]}</div>{framework_html}</div>'
        )
    return f'<div class="itd-rec-row">{"".join(cards)}</div>'


def pipeline_diagram(stages: list[str], active_index: int | None = None) -> str:
    parts = []
    for i, stage in enumerate(stages):
        if i > 0:
            parts.append('<span class="itd-pipeline-arrow">&rarr;</span>')
        cls = "itd-pipeline-stage active" if active_index is not None and i <= active_index else "itd-pipeline-stage"
        parts.append(f'<div class="{cls}">{stage}</div>')
    return f'<div class="itd-pipeline">{"".join(parts)}</div>'


def campaign_card(inner_html: str, tone: str | None = None) -> str:
    style = f' style="--itd-tone:{tone};"' if tone else ""
    return f'<div class="itd-campaign-card"{style}>{inner_html}</div>'


def scope_label(text: str, live: bool = False) -> str:
    """A small eyebrow label marking a group of metrics as either "live"
    (recomputed from the current sidebar filters) or fixed/run-level --
    the two look identical today, which the audit flagged as a real
    source of analyst confusion (a KPI that silently ignores the filters
    you just set looks exactly like one that doesn't).
    """
    cls = "itd-scope-label live" if live else "itd-scope-label"
    return f'<div class="{cls}"><span class="itd-scope-dot"></span>{text}</div>'


def disclaimer_banner(text: str) -> str:
    return f'<div class="itd-disclaimer"><b>Synthetic data notice:</b> {text}</div>'
