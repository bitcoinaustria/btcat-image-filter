import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, MATCH, ALL, ctx
from btcat_images.constants import BRANDS, DitherPattern
from typing import Literal, get_args
import base64
import io
from PIL import Image
from btcat_images.core.pipeline import apply_dither
import plotly.graph_objects as go

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define basic layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("BTCAT Image Dithering UI", className="text-center my-4"), width=12)
    ]),

    # Main Content
    dbc.Row([
        # Sidebar for Controls
        dbc.Col(
            html.Div(id='sidebar', className="p-3 bg-light rounded", children=[
                html.H4("Controls"),
                html.Hr(),
                # Sidebar controls will go here
                # Image Upload
                dbc.Card([
                    dbc.CardHeader("Image"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                    ])
                ], className="mb-3"),

                # Global Settings
                dbc.Card([
                    dbc.CardHeader("Global Settings"),
                    dbc.CardBody([
                        dbc.Label("Pattern"),
                        dcc.Dropdown(
                            id='input-pattern',
                            options=[{'label': p.title(), 'value': p} for p in get_args(DitherPattern)],
                            value='floyd-steinberg',
                            clearable=False,
                            className="mb-2"
                        ),

                        dbc.Label("Brand / Color"),
                        dcc.Dropdown(
                            id='input-brand',
                            options=[{'label': b.upper(), 'value': b} for b in BRANDS.keys()],
                            value='btcat',
                            clearable=False,
                            className="mb-2"
                        ),

                        dbc.Label("Background"),
                        dcc.Dropdown(
                            id='input-background',
                            options=[
                                {'label': 'White', 'value': 'white'},
                                {'label': 'Dark', 'value': 'dark'}
                            ],
                            value='white',
                            clearable=False,
                            className="mb-2"
                        ),

                        dbc.Label("Cut Direction"),
                        dcc.Dropdown(
                            id='input-cut',
                            options=[
                                {'label': 'Vertical', 'value': 'vertical'},
                                {'label': 'Horizontal', 'value': 'horizontal'}
                            ],
                            value='vertical',
                            clearable=False,
                            className="mb-2"
                        ),

                        dbc.Label("Cut Position"),
                        dcc.Slider(
                            id='input-pos',
                            min=0.0, max=1.0, step=0.01,
                            value=0.382,
                            marks={0: '0%', 0.5: '50%', 1: '100%'},
                            className="mb-3"
                        ),

                        dbc.Row([
                            dbc.Col(dbc.Checklist(
                                options=[{"label": "Grayscale Original", "value": "grayscale"}],
                                value=["grayscale"],
                                id="input-grayscale",
                                switch=True,
                            ))
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(dbc.Checklist(
                                options=[{"label": "Satoshi Mode", "value": "satoshi"}],
                                value=[],
                                id="input-satoshi",
                                switch=True,
                            ))
                        ], className="mb-2"),
                    ])
                ], className="mb-3"),

                # Tuning Settings
                dbc.Card([
                    dbc.CardHeader("Tuning / Effects"),
                    dbc.CardBody([
                        dbc.Label("Fade / Density"),
                        dcc.Slider(
                            id='input-fade',
                            min=0.0, max=1.0, step=0.05,
                            value=1.0,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            className="mb-3"
                        ),

                        dbc.Label("Jitter"),
                        dcc.Slider(
                            id='input-jitter',
                            min=0.0, max=100.0, step=1.0,
                            value=15.0,
                            marks={0: '0', 50: '50', 100: '100'},
                            className="mb-3"
                        ),

                        dbc.Label("Glitch Amount"),
                        dcc.Slider(
                            id='input-glitch',
                            min=0.0, max=1.0, step=0.01,
                            value=0.0,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            className="mb-3"
                        ),

                        dbc.Label("Shade (e.g. 1 or 0.5,q=3)"),
                        dbc.Input(id="input-shade", value="1", type="text", className="mb-3"),

                        dbc.Label("Blue Noise Amount"),
                        dcc.Slider(
                            id='input-blue-noise',
                            min=0.0, max=1.0, step=0.01,
                            value=0.0,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            className="mb-3"
                        ),

                        dbc.Label("Seed"),
                        dcc.Slider(
                            id='input-seed',
                            min=0, max=2121, step=1,
                            value=0,
                            marks={0: '0', 2121: '2121'},
                            className="mb-3"
                        ),
                    ])
                ], className="mb-3"),

                # Masks Section
                dbc.Card([
                    dbc.CardHeader("Masking Areas"),
                    dbc.CardBody([
                        dbc.ButtonGroup([
                            dbc.Button("Add Rectangle", id="btn-add-rect", color="primary", size="sm"),
                            dbc.Button("Add Circle", id="btn-add-circle", color="secondary", size="sm"),
                        ], className="mb-3 w-100"),
                        html.Div(id="masks-container", children=[])
                    ])
                ], className="mb-3"),

                html.Div(id="controls-container")
            ]),
            style={"maxHeight": "100vh", "overflowY": "auto"},
            width=3
        ),

        # Main Area for Image Preview
        dbc.Col(
            html.Div(id='main-area', className="p-3 bg-white rounded shadow-sm", style={"height": "88vh", "display": "flex", "flexDirection": "column"}, children=[
                dbc.Row([
                    dbc.Col(html.H4("Preview"), width=8),
                    dbc.Col(dbc.Checklist(
                        options=[{"label": "Show Original", "value": "original"}],
                        value=[],
                        id="switch-original",
                        switch=True,
                        inline=True,
                    ), width=2, className="d-flex align-items-center justify-content-end"),
                    dbc.Col(dbc.Button("Download", id="btn-download", color="success", className="w-100"), width=2)
                ], className="mb-2"),
                dcc.Download(id="download-image"),

                # We need a dcc.Store to hold the base64 encoded strings
                dcc.Store(id='store-image-original'),
                dcc.Store(id='store-image-processed'),
                dcc.Store(id='store-image-filename'),

                html.Div(
                    dcc.Graph(
                        id='image-graph',
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'hoverCompareCartesian', 'hoverClosestCartesian']
                        },
                        style={"height": "100%", "width": "100%"}
                    ),
                    style={"flex": 1, "minHeight": 0, "border": "1px solid #dee2e6", "borderRadius": "4px", "overflow": "hidden"}
                ),

                # Command Line Preview
                html.Div([
                    html.H6("CLI Equivalent", className="mt-3 mb-2"),
                    dbc.InputGroup([
                        dbc.Input(id="cli-command", readonly=True, className="text-muted font-monospace bg-light"),
                        dbc.Button("Copy", id="btn-copy-cli", color="secondary", outline=True, n_clicks=0),
                        dcc.Clipboard(id="clipboard-cli", style={"display": "none"})
                    ])
                ], className="mt-auto")
            ]),
            width=9
        )
    ])
], fluid=True, className="p-4 bg-light")


@app.callback(
    Output('masks-container', 'children'),
    Input('btn-add-rect', 'n_clicks'),
    Input('btn-add-circle', 'n_clicks'),
    Input({'type': 'btn-remove-mask', 'index': ALL}, 'n_clicks'),
    State('masks-container', 'children')
)
def manage_masks(n_rect, n_circle, n_removes, children):
    button_id = ctx.triggered_id
    if not button_id:
        return children

    # Initialize if None
    if children is None:
        children = []

    # Count to ensure unique IDs
    import time
    new_index = str(time.time())

    if button_id == 'btn-add-rect':
        new_mask = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H6("Rectangle", className="m-0"), width=9),
                    dbc.Col(dbc.Button("X", id={'type': 'btn-remove-mask', 'index': new_index}, color="danger", size="sm", className="float-end"), width=3)
                ], className="mb-2 align-items-center"),
                dbc.Row([
                    dbc.Col(dbc.Input(id={'type': 'rect-input', 'index': new_index}, placeholder="x1, y1, x2, y2", value="0.5, 0, 1, 1", size="sm"))
                ])
            ], className="p-2")
        ], id={'type': 'mask-card', 'index': new_index}, className="mb-2 shadow-sm")
        children.append(new_mask)
    elif button_id == 'btn-add-circle':
        new_mask = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H6("Circle", className="m-0"), width=9),
                    dbc.Col(dbc.Button("X", id={'type': 'btn-remove-mask', 'index': new_index}, color="danger", size="sm", className="float-end"), width=3)
                ], className="mb-2 align-items-center"),
                dbc.Row([
                    dbc.Col(dbc.Input(id={'type': 'circle-input', 'index': new_index}, placeholder="cx, cy, r", value="0.5, 0.5, 0.3", size="sm"))
                ])
            ], className="p-2")
        ], id={'type': 'mask-card', 'index': new_index}, className="mb-2 shadow-sm")
        children.append(new_mask)
    elif isinstance(button_id, dict) and button_id.get('type') == 'btn-remove-mask':
        remove_index = button_id.get('index')
        children = [c for c in children if c['props']['id']['index'] != remove_index]

    return children

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    return img

@app.callback(
    Output('store-image-original', 'data'),
    Output('store-image-filename', 'data'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_uploaded_image(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update

    # Simply store the full original dataURI to not lose quality
    # This might be large, but it's local.
    return contents, filename

def pil_to_b64(img: Image.Image, format="WEBP"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/webp;base64,{img_str}"

def create_figure(img_src):
    fig = go.Figure()
    # Add invisible trace to define bounds for panning
    fig.add_trace(go.Image(source=img_src))

    # Hide axes and set layout
    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white",
        dragmode='pan',
        # Keep aspect ratio
        yaxis_scaleanchor="x"
    )
    return fig

@app.callback(
    Output('input-cut', 'disabled'),
    Output('input-pos', 'disabled'),
    Input('masks-container', 'children')
)
def disable_cut_controls(children):
    # Disable cut controls if any custom masks exist
    has_masks = children is not None and len(children) > 0
    return has_masks, has_masks


@app.callback(
    Output('store-image-processed', 'data'),
    Input('store-image-original', 'data'),
    Input('input-pattern', 'value'),
    Input('input-brand', 'value'),
    Input('input-background', 'value'),
    Input('input-cut', 'value'),
    Input('input-pos', 'value'),
    Input('input-grayscale', 'value'),
    Input('input-satoshi', 'value'),
    Input('input-fade', 'value'),
    Input('input-jitter', 'value'),
    Input('input-glitch', 'value'),
    Input('input-shade', 'value'),
    Input('input-blue-noise', 'value'),
    Input('input-seed', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def process_image(original_b64, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade, blue_noise, seed, rect_inputs, circle_inputs):
    if not original_b64:
        return dash.no_update

    try:
        img = parse_contents(original_b64)

        rectangles = []
        for r in rect_inputs:
            if r:
                parts = [float(x.strip()) for x in r.split(',')]
                if len(parts) == 4:
                    rectangles.append((parts[0], parts[1], parts[2], parts[3]))

        circles = []
        for c in circle_inputs:
            if c:
                parts = [float(x.strip()) for x in c.split(',')]
                if len(parts) == 3:
                    circles.append((parts[0], parts[1], parts[2]))

        grayscale_original = 'grayscale' in grayscale_list
        satoshi_mode = 'satoshi' in satoshi_list

        processed_img = apply_dither(
            img=img,
            split_ratio=pos,
            cut_direction=cut,
            grayscale_original=grayscale_original,
            randomize=True,
            jitter=jitter,
            rectangles=rectangles,
            circles=circles,
            fade=fade,
            background=background,
            pattern=pattern,
            satoshi_mode=satoshi_mode,
            brand=brand,
            glitch=glitch,
            shade=shade,
            blue_noise=blue_noise,
            seed=seed if seed > 0 else None
        )

        return pil_to_b64(processed_img)
    except Exception as e:
        print(f"Error processing image: {e}")
        return dash.no_update

@app.callback(
    Output('image-graph', 'figure'),
    Input('store-image-processed', 'data'),
    Input('store-image-original', 'data'),
    Input('switch-original', 'value')
)
def update_graph(processed_b64, original_b64, switch_val):
    if not processed_b64:
        if original_b64:
            return create_figure(original_b64)
        return dash.no_update

    show_original = 'original' in (switch_val or [])

    if show_original and original_b64:
        return create_figure(original_b64)

    return create_figure(processed_b64)

@app.callback(
    Output("clipboard-cli", "content"),
    Input("btn-copy-cli", "n_clicks"),
    State("cli-command", "value"),
    prevent_initial_call=True
)
def copy_cli(n, value):
    return value

@app.callback(
    Output("cli-command", "value"),
    Input('store-image-filename', 'data'),
    Input('input-pattern', 'value'),
    Input('input-brand', 'value'),
    Input('input-background', 'value'),
    Input('input-cut', 'value'),
    Input('input-pos', 'value'),
    Input('input-grayscale', 'value'),
    Input('input-satoshi', 'value'),
    Input('input-fade', 'value'),
    Input('input-jitter', 'value'),
    Input('input-glitch', 'value'),
    Input('input-shade', 'value'),
    Input('input-blue-noise', 'value'),
    Input('input-seed', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def update_cli_command(filename, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade, blue_noise, seed, rect_inputs, circle_inputs):
    parts = ["uv run btcat-dither"]

    has_shapes = False

    if rect_inputs:
        for r in rect_inputs:
            if r: parts.append(f'--rect="{r.replace(" ", "")}"')
            has_shapes = True

    if circle_inputs:
        for c in circle_inputs:
            if c: parts.append(f'--circle="{c.replace(" ", "")}"')
            has_shapes = True

    if not has_shapes:
        if cut != 'vertical': parts.append(f'--cut={cut}')
        if pos != 0.382: parts.append(f'--pos={pos}')

    if pattern != 'floyd-steinberg': parts.append(f'--pattern={pattern}')
    if brand != 'btcat': parts.append(f'--brand={brand}')
    if background != 'white': parts.append(f'--background={background}')
    if fade != 1.0: parts.append(f'--fade={fade}')
    if jitter != 15.0: parts.append(f'--jitter={jitter}')
    if glitch > 0.0: parts.append(f'--glitch={glitch}')
    if shade != '1': parts.append(f'--shade="{shade}"')
    if blue_noise > 0.0: parts.append(f'--blue-noise={blue_noise}')
    if seed > 0: parts.append(f'--seed={seed}')

    if 'grayscale' in grayscale_list: parts.append('--grayscale')
    if 'satoshi' in satoshi_list: parts.append('--satoshi-mode')

    img_name = filename if filename else 'image.jpg'
    parts.append(f'"{img_name}"')

    return " ".join(parts)


@app.callback(
    Output("download-image", "data"),
    Input("btn-download", "n_clicks"),
    State("store-image-processed", "data"),
    State("store-image-filename", "data"),
    State("input-pattern", "value"),
    State("input-brand", "value"),
    prevent_initial_call=True
)
def download_image(n_clicks, processed_b64, original_filename, pattern, brand):
    if not processed_b64:
        return dash.no_update

    # Generate download filename
    import os
    base = "image"
    if original_filename:
        base, _ = os.path.splitext(original_filename)

    download_filename = f"{base}-{brand}-{pattern}-dither.webp"

    # Decode base64 to bytes
    content_type, content_string = processed_b64.split(',')
    decoded = base64.b64decode(content_string)

    return dcc.send_bytes(decoded, download_filename)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
