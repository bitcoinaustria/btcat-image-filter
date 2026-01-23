import sys
import click
from typing import Optional, Literal
from .constants import BRANDS, DitherPattern
from .core.pipeline import dither_image

@click.command()
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--cut',
    type=click.Choice(['vertical', 'horizontal'], case_sensitive=False),
    default='vertical',
    show_default=True,
    help='Cut direction: vertical (left/right) or horizontal (top/bottom)'
)
@click.option(
    '--pos',
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='Position of the cut (0.0 to 1.0). Default: golden ratio ~0.382'
)
@click.option(
    '--threshold',
    type=click.IntRange(0, 255),
    default=128,
    show_default=True,
    help='Threshold for dithering (0-255)'
)
@click.option(
    '--pattern',
    type=click.Choice(['floyd-steinberg', 'ordered', 'atkinson', 'clustered-dot', 'bitcoin', 'hal'], case_sensitive=False),
    default='floyd-steinberg',
    show_default=True,
    help='Dithering pattern to use.'
)
@click.option(
    '--grayscale',
    is_flag=True,
    help='Convert entire image to grayscale before applying dithering'
)
@click.option(
    '--no-randomize',
    is_flag=True,
    help='Disable threshold randomization (produces more regular patterns)'
)
@click.option(
    '--jitter',
    type=click.FloatRange(0.0, 255.0),
    default=30.0,
    show_default=True,
    help='Amount of random noise to add to threshold.'
)
@click.option(
    '--reference-width',
    type=int,
    default=1024,
    show_default=True,
    help='Reference width for point size scaling. Dithering point size will be proportional to image width relative to this.'
)
@click.option(
    '--darkness',
    type=float,
    default=0.0,
    show_default=True,
    help='Adjust darkness (draws less background pixels). Positive values make it darker.'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducible results.',
    hidden=False
)
@click.option(
    '--output', '-o',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Output file path. Defaults to automatic naming.'
)
@click.option(
    '--rect',
    multiple=True,
    help='Rectangle region: x1,y1,x2,y2 (fractions, can be outside 0-1). Can be specified multiple times.'
)
@click.option(
    '--circle',
    multiple=True,
    help='Circle region: x,y,radius (fractions, can be outside 0-1). Can be specified multiple times.'
)
@click.option(
    '--fade',
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help='Dithering density (0.0 to 1.0). E.g., 0.1 = only 10%% of pixels dithered (sparse effect). Applies to all dithered areas.'
)
@click.option(
    '--gradient',
    default=None,
    help='Gradient density: angle,start,end (e.g., "0,0.1,1.0"). Angle: 0-360° (0=left→right, 90=top→bottom, 180=right→left, 270=bottom→top). Start/End: density 0.0-1.0. Overrides --fade.'
)
@click.option(
    '--background',
    type=click.Choice(['white', 'dark'], case_sensitive=False),
    default='white',
    show_default=True,
    help='Background color for dithered areas. "white" for white background, "dark" for dark gray (#222222).'
)
@click.option(
    '--satoshi-mode',
    is_flag=True,
    help='Enable Satoshi Mode: Dynamic threshold based on local brightness.'
)
@click.option(
    '--brand',
    type=click.Choice(list(BRANDS.keys()), case_sensitive=False),
    default='btcat',
    show_default=True,
    help='Brand palette to use.'
)
@click.option(
    '--glitch',
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    help='Glitch Mode intensity (0.0 to 1.0). Adds row swapping, channel shifting, and repeated passes.'
)
@click.option(
    '--shade',
    type=str,
    default="1",
    show_default=True,
    help='Shade factor and quantization (e.g. "1", "0.5", "0.5,q=3"). Controls intensity of dot shading based on grayscale value.'
)
def main(
    image: str,
    cut: Literal['vertical', 'horizontal'],
    pos: Optional[float],
    threshold: int,
    pattern: DitherPattern,
    grayscale: bool,
    no_randomize: bool,
    jitter: float,
    reference_width: int,
    darkness: float,
    seed: Optional[int],
    output: Optional[str],
    rect: tuple[str, ...],
    circle: tuple[str, ...],
    fade: Optional[float],
    gradient: Optional[str],
    background: Literal['white', 'dark'],
    satoshi_mode: bool,
    brand: str,
    glitch: float,
    shade: str
) -> None:
    """Apply monochrome dithering to a portion of an image using brand colors.

    IMAGE is the path to the input image file (PNG or JPG).

    By default, randomization is applied to the dithering threshold to create
    more organic, less regular patterns. Use --no-randomize for classic
    Floyd-Steinberg without randomization.

    Dithering modes:

    - Default: Vertical cut at golden ratio (can be changed with --cut and --pos)

    - Rectangles: Specify one or more with --rect=x1,y1,x2,y2
      Example: --rect=0,0,0.1,1 --rect=0.9,0,1,1 (left and right strips)

    - Circles: Specify one or more with --circle=x,y,radius
      Example: --circle=0.5,0.5,0.2 (centered circle)

    - Mix: Combine rectangles and circles
      Example: --rect=0,0,1,0.1 --circle=0.5,0.5,0.2

    Options apply globally:
    - --pattern: Choose algorithm (floyd-steinberg, ordered, atkinson, clustered-dot, bitcoin, hal)
    - --grayscale: Converts entire image to grayscale before applying dithering
    - --fade: Controls dithering density (0.1 = sparse/10%, 1.0 = full density)
    - --satoshi-mode: Adapts threshold per pixel based on local brightness
    - --glitch: Adds glitch effects (row swapping, channel shift, repeated passes)
    """
    try:
        # Parse rectangle specifications
        rectangles: Optional[list[tuple[float, float, float, float]]] = None
        if rect:
            rectangles = []
            for r in rect:
                try:
                    parts = [float(x.strip()) for x in r.split(',')]
                    if len(parts) != 4:
                        raise ValueError(f"Rectangle must have 4 values (x1,y1,x2,y2), got {len(parts)}")
                    rectangles.append((parts[0], parts[1], parts[2], parts[3]))
                except ValueError as e:
                    click.secho(f"Error parsing rectangle '{r}': {e}", fg='red', err=True)
                    sys.exit(1)

        # Parse circle specifications
        circles: Optional[list[tuple[float, float, float]]] = None
        if circle:
            circles = []
            for c in circle:
                try:
                    parts = [float(x.strip()) for x in c.split(',')]
                    if len(parts) != 3:
                        raise ValueError(f"Circle must have 3 values (x,y,radius), got {len(parts)}")
                    circles.append((parts[0], parts[1], parts[2]))
                except ValueError as e:
                    click.secho(f"Error parsing circle '{c}': {e}", fg='red', err=True)
                    sys.exit(1)

        # Parse gradient specification
        gradient_tuple: Optional[tuple[float, float, float]] = None
        if gradient:
            try:
                parts = [float(x.strip()) for x in gradient.split(',')]
                if len(parts) != 3:
                    raise ValueError(f"Gradient must have 3 values (angle,start,end), got {len(parts)}")
                angle, grad_start, grad_end = parts[0], parts[1], parts[2]

                # Validate ranges
                if not (0.0 <= grad_start <= 1.0):
                    raise ValueError(f"Gradient start must be between 0.0 and 1.0, got {grad_start}")
                if not (0.0 <= grad_end <= 1.0):
                    raise ValueError(f"Gradient end must be between 0.0 and 1.0, got {grad_end}")

                gradient_tuple = (angle, grad_start, grad_end)
            except ValueError as e:
                click.secho(f"Error parsing gradient '{gradient}': {e}", fg='red', err=True)
                sys.exit(1)

        output_path = dither_image(
            image,
            split_ratio=pos,
            cut_direction=cut,
            threshold=threshold,
            grayscale_original=grayscale,
            randomize=not no_randomize,
            jitter=jitter,
            reference_width=reference_width,
            darkness_offset=darkness,
            seed=seed,
            output_path=output,
            rectangles=rectangles,
            circles=circles,
            fade=fade,
            gradient=gradient_tuple,
            background=background,
            pattern=pattern,
            satoshi_mode=satoshi_mode,
            brand=brand,
            glitch=glitch,
            shade=shade
        )
        click.secho(f"✓ Dithered image saved to: {output_path}", fg='green')
    except Exception as e:
        click.secho(f"Error: {e}", fg='red', err=True)
        sys.exit(1)
