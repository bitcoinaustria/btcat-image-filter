"""
btcat Image Dither Explorer — Dash Web Interface

Run with:
    uv run python web_app.py
    (or use web.sh)

Then open: http://localhost:8050
"""

import base64
import io
import re
import time
from pathlib import Path
from typing import Optional

import dash
from dash import (
    ALL,
    Input,
    Output,
    State,
    clientside_callback,
    ctx,
    dcc,
    html,
    no_update,
)
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from btcat_images.constants import BRANDS
from btcat_images.core.pipeline import apply_dither

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="btcat Dither Explorer",
    suppress_callback_exceptions=True,
)
server = app.server

# Inject CSS for pixel-accurate image rendering inside the Plotly graph
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        /* Pixel-crisp rendering for the dithered image inside Plotly */
        .main-svg image { image-rendering: pixelated; }
        /* Grab cursor when panning is active */
        .nsewdrag { cursor: grab !important; }
        .nsewdrag:active { cursor: grabbing !important; }
        /* Compact accordion */
        .accordion-button { padding: 8px 12px !important; font-size: 0.82rem !important; }
        .accordion-body  { padding: 10px 12px !important; }
        /* Thin scrollbar for sidebar */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
        /* Overlay loading spinner on top of graph */
        ._dash-loading { position: absolute; z-index: 100; }
    </style>
</head>
<body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>"""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    """Encode PIL image as base64 WebP."""
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def make_figure(img: Image.Image, reset_zoom: bool = True) -> go.Figure:
    """Wrap a PIL Image in a Plotly figure with pan/zoom support."""
    w, h = img.size
    fig = px.imshow(img)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        dragmode="pan",
        hovermode=False,
        coloraxis_showscale=False,
        # Keep zoom/pan when figure data updates (toggle before/after).
        # Set to unique value only when we want a forced reset.
        uirevision="stable" if not reset_zoom else str(time.time()),
    )
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False,
        range=[0, w], constrain="domain",
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False,
        range=[h, 0], scaleanchor="x",
    )
    fig.update_traces(hovertemplate=None)
    return fig


def make_empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.add_annotation(
        text="Upload an image to start",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=18, color="#555"),
    )
    return fig


def encode_filename(
    base_name: str,
    grayscale: bool,
    pattern: str,
    brand: str,
    threshold: int,
    fade: Optional[float],
    gradient_enabled: bool,
    g_angle: float,
    g_start: float,
    g_end: float,
    masks: list,
) -> str:
    """Build a descriptive download filename that encodes key params."""
    safe = re.sub(r"[^\w-]", "", base_name) or "image"
    parts = [safe, "dither"]
    if grayscale:
        parts.append("gray")
    parts.append(pattern[:2].upper())
    if brand != "btcat":
        parts.append(brand[:3])
    parts.append(f"t{threshold}")
    if fade is not None:
        parts.append(f"f{fade:.1f}")
    if gradient_enabled:
        parts.append(f"g{int(g_angle)}-{g_start:.1f}-{g_end:.1f}")
    for m in masks[:4]:           # cap at 4 shapes in filename
        if m["type"] == "rect":
            parts.append(
                f"R{m['x1']:.2f}-{m['y1']:.2f}-{m['x2']:.2f}-{m['y2']:.2f}"
            )
        else:
            parts.append(f"C{m['cx']:.2f}-{m['cy']:.2f}-{m['r']:.2f}")
    return "-".join(parts) + ".webp"


# ──────────────────────────────────────────────────────────────────────────────
# UI building blocks
# ──────────────────────────────────────────────────────────────────────────────

def slider_row(
    label: str, id: str, min: float, max: float, value: float, step: float = 1.0
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Small(label, className="text-muted"),
                    html.Small(
                        str(value),
                        id={"type": "sval", "index": id},
                        className="text-secondary ms-1",
                    ),
                ],
                className="d-flex justify-content-between",
            ),
            dcc.Slider(
                id=id, min=min, max=max, value=value, step=step,
                marks=None, tooltip=None, className="my-1",
            ),
        ],
        className="mb-2",
    )


def select_row(label: str, comp_id: str, options: list, value: str) -> html.Div:
    return html.Div(
        [
            html.Small(label, className="text-muted d-block"),
            dbc.Select(
                id=comp_id,
                options=[{"label": o, "value": o} for o in options],
                value=value,
                size="sm",
                className="mb-2",
            ),
        ]
    )


def switch_row(label: str, comp_id: str, value: bool = False) -> dbc.Switch:
    return dbc.Switch(id=comp_id, label=label, value=value, className="mb-1 small")


def num_input(comp_id: str, placeholder: str = "", value=None, **kwargs) -> dcc.Input:
    return dcc.Input(
        id=comp_id, type="number", placeholder=placeholder, value=value,
        debounce=True, className="form-control form-control-sm mb-2",
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mask item renderer
# ──────────────────────────────────────────────────────────────────────────────

def make_mask_item(mask: dict) -> dbc.Card:
    mid = mask["id"]
    mtype = mask["type"]

    def mp_input(field, val, **kw):
        return dcc.Input(
            id={"type": "mp", "index": f"{mid}-{field}"},
            type="number", value=val, step=0.01, debounce=True,
            className="form-control form-control-sm",
            **kw,
        )

    if mtype == "rect":
        icon, label = "▭", "Rectangle"
        body = dbc.Row(
            [
                dbc.Col([html.Small("x1"), mp_input("x1", mask.get("x1", 0.5), min=-2, max=2)], width=3),
                dbc.Col([html.Small("y1"), mp_input("y1", mask.get("y1", 0.0), min=-2, max=2)], width=3),
                dbc.Col([html.Small("x2"), mp_input("x2", mask.get("x2", 1.0), min=-2, max=2)], width=3),
                dbc.Col([html.Small("y2"), mp_input("y2", mask.get("y2", 1.0), min=-2, max=2)], width=3),
            ],
            className="g-1",
        )
    else:
        icon, label = "◯", "Circle"
        body = dbc.Row(
            [
                dbc.Col([html.Small("cx"), mp_input("cx", mask.get("cx", 0.5), min=-2, max=2)], width=4),
                dbc.Col([html.Small("cy"), mp_input("cy", mask.get("cy", 0.5), min=-2, max=2)], width=4),
                dbc.Col([html.Small("r"),  mp_input("r",  mask.get("r",  0.2), min=0, max=3)],  width=4),
            ],
            className="g-1",
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span(f"{icon} {label}", className="small fw-bold"),
                    dbc.Button(
                        "✕",
                        id={"type": "mask-remove", "index": mid},
                        color="danger", size="sm",
                        className="ms-auto py-0 px-1",
                        style={"lineHeight": "1.4"},
                    ),
                ],
                className="d-flex align-items-center py-1 px-2",
                style={"background": "#2a2a2a"},
            ),
            dbc.CardBody(body, className="p-2"),
        ],
        className="mb-1",
        style={"border": "1px solid #444", "borderRadius": "4px", "fontSize": "0.78rem"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

SIDEBAR = {
    "width": "350px", "minWidth": "350px",
    "overflowY": "auto", "height": "100vh",
    "padding": "14px", "borderRight": "1px solid #2a2a2a",
    "background": "#161616",
}
PREVIEW_AREA = {
    "flex": "1", "display": "flex", "flexDirection": "column",
    "height": "100vh", "overflow": "hidden", "background": "#111111",
    "position": "relative",
}

GOLDEN = round(1.0 / 1.618033988749895, 3)

app.layout = html.Div(
    [
        # ── Stores ──
        dcc.Store(id="store-original"),      # base64 WebP of uploaded image
        dcc.Store(id="store-processed"),     # base64 WebP of processed result
        dcc.Store(id="store-filename"),      # original filename stem
        dcc.Store(id="store-masks", data={"counter": 0, "masks": []}),

        # ── Root flex container ──
        html.Div(
            [
                # ════════════════ SIDEBAR ════════════════
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("btcat", className="text-danger fw-bold"),
                                html.Span(" Dither Explorer", className="text-light fw-bold"),
                            ],
                            className="mb-3",
                            style={"fontSize": "1.05rem", "letterSpacing": "0.5px"},
                        ),

                        # Upload
                        dcc.Upload(
                            id="upload",
                            children=html.Div(
                                [
                                    html.Div("📁 Drop image here", className="text-center"),
                                    html.Small("or click to browse", className="text-muted d-block text-center"),
                                ]
                            ),
                            accept="image/*",
                            style={
                                "border": "2px dashed #444", "borderRadius": "8px",
                                "padding": "14px", "cursor": "pointer", "marginBottom": "8px",
                            },
                        ),
                        html.Div(id="upload-status", className="text-muted small mb-2"),

                        # ── Accordion sections ──
                        dbc.Accordion(
                            [
                                # — Style —
                                dbc.AccordionItem(
                                    [
                                        select_row("Pattern", "opt-pattern",
                                                   ["floyd-steinberg", "ordered", "atkinson",
                                                    "clustered-dot", "bitcoin", "hal"],
                                                   "floyd-steinberg"),
                                        select_row("Brand", "opt-brand", list(BRANDS.keys()), "btcat"),
                                        select_row("Background", "opt-background", ["white", "dark"], "white"),
                                        select_row("Mode", "opt-mode", ["default", "original"], "default"),
                                        switch_row("Grayscale", "opt-grayscale"),
                                        switch_row("Satoshi Mode", "opt-satoshi"),
                                        switch_row("Randomize threshold", "opt-randomize", value=True),
                                    ],
                                    title="Style",
                                ),

                                # — Adjustments —
                                dbc.AccordionItem(
                                    [
                                        slider_row("Threshold", "opt-threshold", 0, 255, 128),
                                        slider_row("Jitter", "opt-jitter", 0.0, 255.0, 30.0, step=1.0),
                                        slider_row("Darkness", "opt-darkness", -100.0, 100.0, 0.0, step=1.0),
                                        slider_row("Glitch", "opt-glitch", 0.0, 1.0, 0.0, step=0.01),
                                        html.Div(
                                            [
                                                html.Small("Reference width", className="text-muted d-block"),
                                                num_input("opt-ref-width", value=1024, min=64, max=8192, step=64),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Small("Seed  (blank = random)", className="text-muted d-block"),
                                                num_input("opt-seed", placeholder="random", min=0),
                                            ]
                                        ),
                                    ],
                                    title="Adjustments",
                                ),

                                # — Fade & Gradient —
                                dbc.AccordionItem(
                                    [
                                        switch_row("Enable uniform fade", "opt-fade-enable"),
                                        slider_row("Fade density", "opt-fade", 0.0, 1.0, 1.0, step=0.01),
                                        html.Hr(className="my-2", style={"borderColor": "#333"}),
                                        switch_row("Enable gradient", "opt-gradient-enable"),
                                        slider_row("Angle (°)", "opt-gradient-angle", 0, 360, 0, step=1),
                                        slider_row("Density start", "opt-gradient-start", 0.0, 1.0, 0.0, step=0.01),
                                        slider_row("Density end", "opt-gradient-end", 0.0, 1.0, 1.0, step=0.01),
                                        html.Small(
                                            "0°=left→right  90°=top→bottom  180°=right→left  270°=bottom→top",
                                            className="text-muted",
                                        ),
                                    ],
                                    title="Fade & Gradient",
                                ),

                                # — Shading —
                                dbc.AccordionItem(
                                    [
                                        slider_row("Shade factor", "opt-shade", 0.0, 2.0, 1.0, step=0.05),
                                        html.Div(
                                            [
                                                html.Small("Shade quantize (blank = off)", className="text-muted d-block"),
                                                num_input("opt-shade-quant", placeholder="e.g. 3", min=2, max=16, step=1),
                                            ]
                                        ),
                                    ],
                                    title="Shading",
                                ),

                                # — Original Mode —
                                dbc.AccordionItem(
                                    [
                                        html.Small(
                                            "These settings only apply when Mode = original.",
                                            className="text-muted d-block mb-2",
                                        ),
                                        slider_row("Point size", "opt-point-size", 1, 8, 1, step=1),
                                        slider_row("Brightness", "opt-brightness", 0.0, 2.0, 1.0, step=0.05),
                                        slider_row("Contrast", "opt-contrast", 0.0, 2.0, 1.0, step=0.05),
                                        slider_row("Detail", "opt-detail", 0.6, 1.0, 1.0, step=0.01),
                                        slider_row("Bloom intensity", "opt-bloom-intensity", 0.0, 1.0, 0.5, step=0.05),
                                        slider_row("Bloom radius", "opt-bloom-radius", 1.0, 150.0, 75.0, step=1.0),
                                    ],
                                    title="Original Mode",
                                ),

                                # — Cut Mode (fallback) —
                                dbc.AccordionItem(
                                    [
                                        select_row("Cut direction", "opt-cut", ["vertical", "horizontal"], "vertical"),
                                        slider_row("Cut position", "opt-pos", 0.0, 1.0, GOLDEN, step=0.001),
                                        html.Small(
                                            "Applied automatically when no mask areas are defined below.",
                                            className="text-muted",
                                        ),
                                    ],
                                    title="Cut Mode (fallback)",
                                ),
                            ],
                            start_collapsed=True,
                            always_open=True,
                            className="mb-3",
                            style={"fontSize": "0.82rem"},
                        ),

                        # ── Mask Areas ──
                        html.Hr(style={"borderColor": "#2a2a2a"}),
                        html.Small(
                            "MASK AREAS",
                            className="text-muted fw-bold d-block mb-2",
                            style={"letterSpacing": "1px"},
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("+ Rectangle", id="btn-add-rect",
                                           color="secondary", size="sm", outline=True),
                                dbc.Button("+ Circle", id="btn-add-circle",
                                           color="secondary", size="sm", outline=True),
                                dbc.Button("Clear all", id="btn-clear-masks",
                                           color="danger", size="sm", outline=True),
                            ],
                            className="mb-2 w-100",
                        ),
                        html.Div(id="masks-container"),

                        # ── Apply ──
                        html.Hr(style={"borderColor": "#2a2a2a"}),
                        dbc.Button(
                            "⚡ Apply Effect",
                            id="btn-apply", color="danger",
                            className="w-100 mb-1", disabled=True,
                        ),
                        html.Div(id="status-msg", className="text-muted small text-center mb-3"),
                    ],
                    style=SIDEBAR,
                ),

                # ════════════════ PREVIEW AREA ════════════════
                html.Div(
                    [
                        # Top bar
                        html.Div(
                            [
                                dbc.RadioItems(
                                    id="preview-toggle",
                                    options=[
                                        {"label": "Original", "value": "original"},
                                        {"label": "Processed", "value": "processed"},
                                    ],
                                    value="original",
                                    inline=True,
                                    className="me-3",
                                    inputClassName="me-1",
                                ),
                                html.Small(id="img-info", className="text-muted me-auto"),
                                dbc.Button(
                                    "⊕ 1:1 pixels", id="btn-zoom-100",
                                    color="secondary", size="sm", outline=True,
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "↺ Fit", id="btn-zoom-fit",
                                    color="secondary", size="sm", outline=True,
                                    className="me-2",
                                ),
                                dbc.Button(
                                    "⬇ Download", id="btn-download",
                                    color="success", size="sm", disabled=True,
                                ),
                                dcc.Download(id="download"),
                            ],
                            className="d-flex align-items-center px-3 py-2",
                            style={"borderBottom": "1px solid #1e1e1e", "background": "#161616", "flexShrink": 0},
                        ),

                        # Graph
                        dcc.Loading(
                            type="dot",
                            color="#E3000F",
                            style={"position": "absolute", "top": "50%", "left": "60%"},
                            children=dcc.Graph(
                                id="preview-graph",
                                figure=make_empty_figure(),
                                style={"flex": "1", "height": "100%"},
                                config={
                                    "scrollZoom": True,
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": [
                                        "select2d", "lasso2d", "autoScale2d",
                                        "toggleSpikelines",
                                    ],
                                    "displaylogo": False,
                                    "toImageButtonOptions": {"format": "png", "scale": 1},
                                },
                                responsive=True,
                            ),
                        ),
                    ],
                    style=PREVIEW_AREA,
                ),
            ],
            style={"display": "flex", "height": "100vh", "overflow": "hidden"},
        ),
    ],
    style={"fontFamily": "'Courier New', monospace", "background": "#111111", "color": "#cccccc"},
)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Upload image ──────────────────────────────────────────────────────────
@app.callback(
    Output("store-original", "data"),
    Output("store-filename", "data"),
    Output("upload-status", "children"),
    Output("btn-apply", "disabled"),
    Output("img-info", "children"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        return no_update, no_update, no_update, True, ""
    _, b64 = contents.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    w, h = img.size
    return (
        pil_to_b64(img),
        Path(filename).stem,
        f"📷 {filename}",
        False,
        f"{w} × {h} px",
    )


# ── 2. Mask store mutations ───────────────────────────────────────────────────
@app.callback(
    Output("store-masks", "data"),
    Input("btn-add-rect", "n_clicks"),
    Input("btn-add-circle", "n_clicks"),
    Input("btn-clear-masks", "n_clicks"),
    Input({"type": "mask-remove", "index": ALL}, "n_clicks"),
    Input({"type": "mp", "index": ALL}, "value"),
    State("store-masks", "data"),
    prevent_initial_call=True,
)
def update_masks(_add_rect, _add_circle, _clear, _removes, _param_vals, data):
    triggered = ctx.triggered_id
    if triggered is None:
        return no_update

    masks: list = data["masks"]
    counter: int = data["counter"]

    if triggered == "btn-add-rect":
        masks.append({"id": counter, "type": "rect",
                       "x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0})
        counter += 1

    elif triggered == "btn-add-circle":
        masks.append({"id": counter, "type": "circle",
                       "cx": 0.5, "cy": 0.5, "r": 0.2})
        counter += 1

    elif triggered == "btn-clear-masks":
        masks = []

    elif isinstance(triggered, dict) and triggered.get("type") == "mask-remove":
        mid = triggered["index"]
        masks = [m for m in masks if m["id"] != mid]

    elif isinstance(triggered, dict) and triggered.get("type") == "mp":
        # e.g. index = "3-x1"
        try:
            mid_str, field = triggered["index"].rsplit("-", 1)
            mid = int(mid_str)
            value = ctx.triggered[0]["value"]
            if value is not None:
                for m in masks:
                    if m["id"] == mid:
                        m[field] = float(value)
                        break
        except (ValueError, IndexError):
            pass

    return {"counter": counter, "masks": masks}


# ── 3. Render mask items ──────────────────────────────────────────────────────
@app.callback(
    Output("masks-container", "children"),
    Input("store-masks", "data"),
)
def render_masks(data):
    return [make_mask_item(m) for m in data["masks"]]


# ── 4. Apply effect ───────────────────────────────────────────────────────────
@app.callback(
    Output("store-processed", "data"),
    Output("status-msg", "children"),
    Input("btn-apply", "n_clicks"),
    State("store-original", "data"),
    State("store-masks", "data"),
    # style
    State("opt-pattern", "value"),
    State("opt-brand", "value"),
    State("opt-background", "value"),
    State("opt-mode", "value"),
    State("opt-grayscale", "value"),
    State("opt-satoshi", "value"),
    State("opt-randomize", "value"),
    # adjustments
    State("opt-threshold", "value"),
    State("opt-jitter", "value"),
    State("opt-darkness", "value"),
    State("opt-glitch", "value"),
    State("opt-ref-width", "value"),
    State("opt-seed", "value"),
    # fade / gradient
    State("opt-fade-enable", "value"),
    State("opt-fade", "value"),
    State("opt-gradient-enable", "value"),
    State("opt-gradient-angle", "value"),
    State("opt-gradient-start", "value"),
    State("opt-gradient-end", "value"),
    # shading
    State("opt-shade", "value"),
    State("opt-shade-quant", "value"),
    # original mode
    State("opt-point-size", "value"),
    State("opt-brightness", "value"),
    State("opt-contrast", "value"),
    State("opt-detail", "value"),
    State("opt-bloom-intensity", "value"),
    State("opt-bloom-radius", "value"),
    # cut fallback
    State("opt-cut", "value"),
    State("opt-pos", "value"),
    prevent_initial_call=True,
)
def apply_effect(
    _n,
    orig_b64,
    masks_data,
    pattern, brand, background, mode,
    grayscale, satoshi, randomize,
    threshold, jitter, darkness, glitch, ref_width, seed,
    fade_enable, fade,
    gradient_enable, g_angle, g_start, g_end,
    shade, shade_quant,
    point_size, brightness, contrast, detail, bloom_intensity, bloom_radius,
    cut, pos,
):
    if not orig_b64:
        return no_update, "⚠ No image uploaded"

    try:
        t0 = time.time()
        img = b64_to_pil(orig_b64)

        # Collect shapes
        rectangles = [
            (m["x1"], m["y1"], m["x2"], m["y2"])
            for m in masks_data["masks"] if m["type"] == "rect"
        ]
        circles = [
            (m["cx"], m["cy"], m["r"])
            for m in masks_data["masks"] if m["type"] == "circle"
        ]

        # Fade / gradient
        fade_val = float(fade) if fade_enable and fade is not None else None
        gradient_val = None
        if gradient_enable and gradient_enable and not fade_enable:
            gradient_val = (float(g_angle), float(g_start), float(g_end))

        # Shade string
        shade_str = str(float(shade)) if shade is not None else "1"
        if shade_quant is not None:
            shade_str += f",q={int(shade_quant)}"

        result = apply_dither(
            img,
            split_ratio=float(pos) if pos is not None else None,
            cut_direction=cut or "vertical",
            threshold=int(threshold) if threshold is not None else 128,
            grayscale_original=bool(grayscale),
            randomize=bool(randomize),
            jitter=float(jitter) if jitter is not None else 30.0,
            reference_width=int(ref_width) if ref_width else 1024,
            darkness_offset=float(darkness) if darkness is not None else 0.0,
            seed=int(seed) if seed is not None else None,
            rectangles=rectangles or None,
            circles=circles or None,
            fade=fade_val,
            gradient=gradient_val,
            background=background or "white",
            pattern=pattern or "floyd-steinberg",
            satoshi_mode=bool(satoshi),
            brand=brand or "btcat",
            glitch=float(glitch) if glitch is not None else 0.0,
            shade=shade_str,
            mode=mode or "default",
            point_size=int(point_size) if point_size else 1,
            brightness=float(brightness) if brightness is not None else 1.0,
            contrast=float(contrast) if contrast is not None else 1.0,
            detail=float(detail) if detail is not None else 1.0,
            bloom_intensity=float(bloom_intensity) if bloom_intensity is not None else 0.5,
            bloom_radius=float(bloom_radius) if bloom_radius is not None else 75.0,
        )

        elapsed = time.time() - t0
        return pil_to_b64(result), f"✓ Done in {elapsed:.1f}s"

    except Exception as exc:
        return no_update, f"✗ {exc}"


# ── 5. Switch to processed tab after a successful apply ───────────────────────
@app.callback(
    Output("preview-toggle", "value"),
    Input("store-processed", "data"),
    prevent_initial_call=True,
)
def switch_to_processed(data):
    return "processed" if data else no_update


# ── 6. Update preview graph ───────────────────────────────────────────────────
@app.callback(
    Output("preview-graph", "figure"),
    Output("btn-download", "disabled"),
    Input("preview-toggle", "value"),
    Input("store-original", "data"),
    Input("store-processed", "data"),
    prevent_initial_call=False,
)
def update_preview(toggle, orig_b64, proc_b64):
    show_b64 = orig_b64 if toggle == "original" else proc_b64
    if not show_b64:
        # Fall back to original when processed is not ready yet
        show_b64 = orig_b64
    if not show_b64:
        return make_empty_figure(), True
    img = b64_to_pil(show_b64)
    # Reset zoom only when a NEW image is uploaded (original changes)
    reset = ctx.triggered_id == "store-original"
    return make_figure(img, reset_zoom=reset), proc_b64 is None


# ── 7. Fit zoom (reset) button ────────────────────────────────────────────────
@app.callback(
    Output("preview-graph", "figure", allow_duplicate=True),
    Input("btn-zoom-fit", "n_clicks"),
    State("preview-toggle", "value"),
    State("store-original", "data"),
    State("store-processed", "data"),
    prevent_initial_call=True,
)
def zoom_fit(_n, toggle, orig_b64, proc_b64):
    show_b64 = orig_b64 if toggle == "original" else proc_b64
    if not show_b64:
        return no_update
    img = b64_to_pil(show_b64)
    return make_figure(img, reset_zoom=True)


# ── 8. 1:1 pixel zoom button (clientside for accuracy) ───────────────────────
# We update the figure's uirevision to force a re-render, then the
# clientside callback adjusts axis ranges to achieve 1:1 pixel mapping.
@app.callback(
    Output("preview-graph", "figure", allow_duplicate=True),
    Input("btn-zoom-100", "n_clicks"),
    State("preview-toggle", "value"),
    State("store-original", "data"),
    State("store-processed", "data"),
    State("preview-graph", "relayoutData"),
    prevent_initial_call=True,
)
def zoom_100(_n, toggle, orig_b64, proc_b64, relayout):
    show_b64 = orig_b64 if toggle == "original" else proc_b64
    if not show_b64:
        return no_update
    img = b64_to_pil(show_b64)
    w, h = img.size
    fig = make_figure(img, reset_zoom=True)
    # Show image centered at 100% — use image native size as axis span.
    # The graph container will clip/scroll; user pans with drag.
    cx, cy = w // 2, h // 2
    fig.update_xaxes(range=[0, w])
    fig.update_yaxes(range=[h, 0])
    return fig


# ── 9. Download ───────────────────────────────────────────────────────────────
@app.callback(
    Output("download", "data"),
    Input("btn-download", "n_clicks"),
    State("store-processed", "data"),
    State("store-filename", "data"),
    State("store-masks", "data"),
    State("opt-pattern", "value"),
    State("opt-brand", "value"),
    State("opt-grayscale", "value"),
    State("opt-threshold", "value"),
    State("opt-fade-enable", "value"),
    State("opt-fade", "value"),
    State("opt-gradient-enable", "value"),
    State("opt-gradient-angle", "value"),
    State("opt-gradient-start", "value"),
    State("opt-gradient-end", "value"),
    prevent_initial_call=True,
)
def download_result(
    _n, proc_b64, stem, masks_data,
    pattern, brand, grayscale, threshold,
    fade_enable, fade,
    gradient_enable, g_angle, g_start, g_end,
):
    if not proc_b64:
        return no_update
    filename = encode_filename(
        stem or "image",
        bool(grayscale),
        pattern or "floyd-steinberg",
        brand or "btcat",
        int(threshold) if threshold is not None else 128,
        float(fade) if fade_enable and fade is not None else None,
        bool(gradient_enable),
        float(g_angle) if g_angle is not None else 0.0,
        float(g_start) if g_start is not None else 0.0,
        float(g_end) if g_end is not None else 1.0,
        masks_data["masks"],
    )
    return dcc.send_bytes(base64.b64decode(proc_b64), filename=filename)


# ── 10. Slider value displays ─────────────────────────────────────────────────
_SLIDER_IDS = [
    "opt-threshold", "opt-jitter", "opt-darkness", "opt-glitch",
    "opt-fade", "opt-gradient-angle", "opt-gradient-start", "opt-gradient-end",
    "opt-shade", "opt-point-size", "opt-brightness", "opt-contrast",
    "opt-detail", "opt-bloom-intensity", "opt-bloom-radius", "opt-pos",
]

@app.callback(
    [Output({"type": "sval", "index": sid}, "children") for sid in _SLIDER_IDS],
    [Input(sid, "value") for sid in _SLIDER_IDS],
)
def update_slider_labels(*vals):
    return [str(v) for v in vals]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\n  btcat Dither Explorer  →  http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
