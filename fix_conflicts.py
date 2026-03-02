import re

with open('webui.py', 'r') as f:
    text = f.read()

# Replace block 1: Shade and Seed
block1_search = """<<<<<<< HEAD
                        dbc.Label("Shade Factor"),
                        dcc.Slider(
                            id='input-shade-factor',
                            min=0.0, max=1.0, step=0.05,
                            value=1.0,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            className="mb-3"
                        ),
                        dbc.Label("Shade Quantization (0 = Off)"),
                        dcc.Slider(
                            id='input-shade-quant',
                            min=0, max=32, step=1,
                            value=4,
                            marks={0: 'Off', 4: '4', 16: '16', 32: '32'},
=======
                        dbc.Label("Shade (e.g. 1 or 0.5,q=3)"),
                        dbc.Input(id="input-shade", value="1", type="text", className="mb-3"),

                        dbc.Label("Seed"),
                        dcc.Slider(
                            id='input-seed',
                            min=0, max=2121, step=1,
                            value=0,
                            marks={0: '0', 2121: '2121'},
>>>>>>> origin/main
                            className="mb-3"
                        ),"""

block1_replace = """                        dbc.Label("Shade Factor"),
                        dcc.Slider(
                            id='input-shade-factor',
                            min=0.0, max=1.0, step=0.05,
                            value=1.0,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            className="mb-3"
                        ),
                        dbc.Label("Shade Quantization (0 = Off)"),
                        dcc.Slider(
                            id='input-shade-quant',
                            min=0, max=32, step=1,
                            value=4,
                            marks={0: 'Off', 4: '4', 16: '16', 32: '32'},
                            className="mb-3"
                        ),

                        dbc.Label("Seed"),
                        dcc.Slider(
                            id='input-seed',
                            min=0, max=2121, step=1,
                            value=0,
                            marks={0: '0', 2121: '2121'},
                            className="mb-3"
                        ),"""

# Replace block 2: process_image inputs
block2_search = """<<<<<<< HEAD
    Input('input-shade-factor', 'value'),
    Input('input-shade-quant', 'value'),
    Input('input-point-size', 'value'),
    Input('input-brightness', 'value'),
    Input('input-contrast', 'value'),
    Input('input-detail', 'value'),
    Input('input-bloom-intensity', 'value'),
    Input('input-bloom-radius', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def process_image(original_b64, mode, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade_factor, shade_quant, point_size, brightness, contrast, detail, bloom_intensity, bloom_radius, rect_inputs, circle_inputs):
=======
    Input('input-shade', 'value'),
    Input('input-seed', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def process_image(original_b64, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade, seed, rect_inputs, circle_inputs):
>>>>>>> origin/main"""

block2_replace = """    Input('input-shade-factor', 'value'),
    Input('input-shade-quant', 'value'),
    Input('input-seed', 'value'),
    Input('input-point-size', 'value'),
    Input('input-brightness', 'value'),
    Input('input-contrast', 'value'),
    Input('input-detail', 'value'),
    Input('input-bloom-intensity', 'value'),
    Input('input-bloom-radius', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def process_image(original_b64, mode, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade_factor, shade_quant, seed, point_size, brightness, contrast, detail, bloom_intensity, bloom_radius, rect_inputs, circle_inputs):"""

# Replace block 3: process_image args
block3_search = """<<<<<<< HEAD
            mode=mode,
            point_size=point_size,
            brightness=brightness,
            contrast=contrast,
            detail=detail,
            bloom_intensity=bloom_intensity,
            bloom_radius=bloom_radius
=======
            seed=seed if seed > 0 else None
>>>>>>> origin/main"""

block3_replace = """            seed=seed if seed > 0 else None,
            mode=mode,
            point_size=point_size,
            brightness=brightness,
            contrast=contrast,
            detail=detail,
            bloom_intensity=bloom_intensity,
            bloom_radius=bloom_radius"""

# Replace block 4: update_cli_command inputs
block4_search = """<<<<<<< HEAD
    Input('input-shade-factor', 'value'),
    Input('input-shade-quant', 'value'),
    Input('input-point-size', 'value'),
    Input('input-brightness', 'value'),
    Input('input-contrast', 'value'),
    Input('input-detail', 'value'),
    Input('input-bloom-intensity', 'value'),
    Input('input-bloom-radius', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def update_cli_command(filename, mode, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade_factor, shade_quant, point_size, brightness, contrast, detail, bloom_intensity, bloom_radius, rect_inputs, circle_inputs):
=======
    Input('input-shade', 'value'),
    Input('input-seed', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def update_cli_command(filename, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade, seed, rect_inputs, circle_inputs):
>>>>>>> origin/main"""

block4_replace = """    Input('input-shade-factor', 'value'),
    Input('input-shade-quant', 'value'),
    Input('input-seed', 'value'),
    Input('input-point-size', 'value'),
    Input('input-brightness', 'value'),
    Input('input-contrast', 'value'),
    Input('input-detail', 'value'),
    Input('input-bloom-intensity', 'value'),
    Input('input-bloom-radius', 'value'),
    Input({'type': 'rect-input', 'index': ALL}, 'value'),
    Input({'type': 'circle-input', 'index': ALL}, 'value')
)
def update_cli_command(filename, mode, pattern, brand, background, cut, pos, grayscale_list, satoshi_list, fade, jitter, glitch, shade_factor, shade_quant, seed, point_size, brightness, contrast, detail, bloom_intensity, bloom_radius, rect_inputs, circle_inputs):"""


# Replace block 5: update_cli_command args
block5_search = """<<<<<<< HEAD
    if mode == 'original':
        parts.append('--mode=original')
        if point_size != 1: parts.append(f'--point-size={point_size}')
        if brightness != 1.0: parts.append(f'--brightness={brightness}')
        if contrast != 1.0: parts.append(f'--contrast={contrast}')
        if detail != 1.0: parts.append(f'--detail={detail}')
        if bloom_intensity != 0.5: parts.append(f'--bloom-intensity={bloom_intensity}')
        if bloom_radius != 75.0: parts.append(f'--bloom-radius={bloom_radius}')
    else:
        if pattern != 'floyd-steinberg': parts.append(f'--pattern={pattern}')
        if brand != 'btcat': parts.append(f'--brand={brand}')
        if background != 'white': parts.append(f'--background={background}')
        if jitter != 15.0: parts.append(f'--jitter={jitter}')

        shade = f"{shade_factor}"
        if shade_quant > 0:
            shade += f",q={shade_quant}"
        if shade != '1.0': parts.append(f'--shade="{shade}"')

        if 'satoshi' in satoshi_list: parts.append('--satoshi-mode')
=======
    if pattern != 'floyd-steinberg': parts.append(f'--pattern={pattern}')
    if brand != 'btcat': parts.append(f'--brand={brand}')
    if background != 'white': parts.append(f'--background={background}')
    if fade != 1.0: parts.append(f'--fade={fade}')
    if jitter != 15.0: parts.append(f'--jitter={jitter}')
    if glitch > 0.0: parts.append(f'--glitch={glitch}')
    if shade != '1': parts.append(f'--shade="{shade}"')
    if seed > 0: parts.append(f'--seed={seed}')
>>>>>>> origin/main"""

block5_replace = """    if mode == 'original':
        parts.append('--mode=original')
        if point_size != 1: parts.append(f'--point-size={point_size}')
        if brightness != 1.0: parts.append(f'--brightness={brightness}')
        if contrast != 1.0: parts.append(f'--contrast={contrast}')
        if detail != 1.0: parts.append(f'--detail={detail}')
        if bloom_intensity != 0.5: parts.append(f'--bloom-intensity={bloom_intensity}')
        if bloom_radius != 75.0: parts.append(f'--bloom-radius={bloom_radius}')
    else:
        if pattern != 'floyd-steinberg': parts.append(f'--pattern={pattern}')
        if brand != 'btcat': parts.append(f'--brand={brand}')
        if background != 'white': parts.append(f'--background={background}')
        if jitter != 15.0: parts.append(f'--jitter={jitter}')

        shade = f"{shade_factor}"
        if shade_quant > 0:
            shade += f",q={shade_quant}"
        if shade != '1.0': parts.append(f'--shade="{shade}"')

        if 'satoshi' in satoshi_list: parts.append('--satoshi-mode')

    if seed > 0: parts.append(f'--seed={seed}')"""

text = text.replace(block1_search, block1_replace)
text = text.replace(block2_search, block2_replace)
text = text.replace(block3_search, block3_replace)
text = text.replace(block4_search, block4_replace)
text = text.replace(block5_search, block5_replace)

with open('webui.py', 'w') as f:
    f.write(text)
