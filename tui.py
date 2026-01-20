#!/usr/bin/env python3
import sys
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Header, Footer, Label, Switch, Select, Input, Static
from textual.binding import Binding
from rich.text import Text
from rich.style import Style
from typing import cast, Literal, Optional, Tuple

# Import core logic from main
try:
    from main import apply_dither, GOLDEN_RATIO
except ImportError:
    # If running from different directory, might need adjustment
    sys.path.append(str(Path(__file__).parent))
    from main import apply_dither, GOLDEN_RATIO

class DitherApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    #sidebar {
        width: 40;
        height: 100%;
        dock: left;
        border-right: solid $accent;
        padding: 1 2;
        background: $surface;
    }
    #preview-container {
        width: 1fr;
        height: 100%;
        align: center middle;
        overflow: auto;
    }
    #preview {
        width: auto;
        height: auto;
    }
    .control-group {
        margin-bottom: 2;
    }
    Label {
        margin-bottom: 1;
        color: $text-muted;
    }
    .header-label {
        color: $text;
        text-style: bold;
        margin-top: 1;
    }
    Input {
        margin-bottom: 1;
    }
    Switch {
        margin-bottom: 1;
    }
    Select {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("o", "open_viewer", "Open Viewer"),
        Binding("s", "save_output", "Save"),
    ]

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        self.original_image = Image.open(self.image_path)
        if self.original_image.mode != 'RGB':
            self.original_image = self.original_image.convert('RGB')

        # Preview file path
        self.preview_path = self.image_path.parent / f"{self.image_path.stem}-preview.png"

        # Debounce timer
        self.update_timer = None

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="sidebar"):
            yield Label("Dithering Controls", classes="header-label")

            yield Label("Cut Direction")
            yield Select.from_values(["vertical", "horizontal"], value="vertical", id="cut_direction")

            yield Label("Split Position (0.0 - 1.0)")
            yield Input(value=str(round(1/GOLDEN_RATIO, 3)), id="pos")

            yield Label("Threshold (0 - 255)")
            yield Input(value="128", id="threshold")

            yield Label("Jitter (0 - 100)")
            yield Input(value="30.0", id="jitter")

            yield Label("Reference Width (Scaling)")
            yield Input(value="1024", id="reference_width")

            yield Label("Darkness Offset (-100 to 100)")
            yield Input(value="0.0", id="darkness")

            yield Label("Fade / Density (0.1 - 1.0)")
            yield Input(value="1.0", id="fade")

            yield Label("Background")
            yield Select.from_values(["white", "dark"], value="white", id="background")

            yield Label("Grayscale Original")
            yield Switch(value=False, id="grayscale")

            yield Label("Randomize Threshold")
            yield Switch(value=True, id="randomize")

            yield Label("Satoshi Mode")
            yield Switch(value=False, id="satoshi_mode")

            yield Label("")
            yield Button("Open External Viewer (o)", id="btn-open", variant="primary")
            yield Label("")
            yield Button("Save & Quit (s)", id="btn-save", variant="success")

        with Container(id="preview-container"):
            yield Static(id="preview")

        yield Footer()

    def on_mount(self):
        self.update_preview()

    def on_input_changed(self, event):
        self.update_preview_debounced()

    def on_switch_changed(self, event):
        self.update_preview()

    def on_select_changed(self, event):
        self.update_preview()

    def on_button_pressed(self, event):
        if event.button.id == "btn-open":
            self.action_open_viewer()
        elif event.button.id == "btn-save":
            self.action_save_output()

    def update_preview_debounced(self):
        if self.update_timer:
            self.update_timer.stop()
        self.update_timer = self.set_timer(0.5, self.update_preview)

    def update_preview(self):
        # Gather parameters
        try:
            cut_val = self.query_one("#cut_direction", Select).value
            # Handle empty selection (Textual Select can be None/BLANK)
            cut_str = str(cut_val) if cut_val != Select.BLANK else "vertical"
            cut: Literal['vertical', 'horizontal'] = cast(Literal['vertical', 'horizontal'], cut_str)

            try:
                pos = float(self.query_one("#pos", Input).value)
            except ValueError:
                pos = 1/GOLDEN_RATIO

            try:
                threshold = int(self.query_one("#threshold", Input).value)
            except ValueError:
                threshold = 128

            try:
                jitter = float(self.query_one("#jitter", Input).value)
            except ValueError:
                jitter = 30.0

            try:
                ref_width = int(self.query_one("#reference_width", Input).value)
            except ValueError:
                ref_width = 1024

            try:
                darkness = float(self.query_one("#darkness", Input).value)
            except ValueError:
                darkness = 0.0

            try:
                fade_val = float(self.query_one("#fade", Input).value)
            except ValueError:
                fade_val = 1.0
            fade = fade_val if fade_val < 1.0 else None

            bg_val = self.query_one("#background", Select).value
            bg_str = str(bg_val) if bg_val != Select.BLANK else "white"
            bg: Literal['white', 'dark'] = cast(Literal['white', 'dark'], bg_str)

            grayscale = self.query_one("#grayscale", Switch).value
            randomize = self.query_one("#randomize", Switch).value
            satoshi = self.query_one("#satoshi_mode", Switch).value

            # Run dithering
            result_img = apply_dither(
                self.original_image,
                split_ratio=pos,
                cut_direction=cut,
                threshold=threshold,
                grayscale_original=grayscale,
                randomize=randomize,
                jitter=jitter,
                reference_width=ref_width,
                darkness_offset=darkness,
                fade=fade,
                background=bg,
                satoshi_mode=satoshi
            )

            # Save preview to file
            result_img.save(self.preview_path)

            # Generate ASCII preview
            ascii_art = self.image_to_ascii(result_img)
            self.query_one("#preview", Static).update(ascii_art)

        except Exception as e:
            # self.notify(f"Error updating preview: {e}", severity="error")
            pass

    def image_to_ascii(self, img: Image.Image) -> Text:
        """Convert PIL image to colored ASCII text for preview."""
        # Calculate resize target
        container = self.query_one("#preview-container")
        width = container.size.width or 80
        height = container.size.height or 40

        # Adjust for padding
        width = max(20, width - 4)
        height = max(10, height - 2)

        img_w, img_h = img.size

        # Determine scaling to fit
        scale_w = width / img_w
        scale_h = (height * 2) / img_h # *2 because we use half-blocks
        scale = min(scale_w, scale_h)

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Ensure new_h is even for half-block rendering
        if new_h % 2 != 0:
            new_h -= 1

        if new_w <= 0 or new_h <= 0:
            return Text("Image too small")

        small_img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
        pixels = small_img.load()

        if pixels is None:
            return Text("Error loading image pixels")

        # Build Text object
        text = Text()

        for y in range(0, new_h, 2):
            for x in range(new_w):
                # Ensure x is within bounds (resize might be slightly off due to float math?)
                # Actually resize result is exact.

                try:
                    r1, g1, b1 = cast(Tuple[int, int, int], pixels[x, y])
                except Exception:
                    r1, g1, b1 = 0, 0, 0

                try:
                    # Check next row if available
                    if y + 1 < new_h:
                        r2, g2, b2 = cast(Tuple[int, int, int], pixels[x, y + 1])
                    else:
                        r2, g2, b2 = 0, 0, 0
                except Exception:
                     r2, g2, b2 = 0, 0, 0

                # Foreground color is top pixel, Background is bottom pixel
                # using unicode upper half block ▀
                color_top = f"rgb({r1},{g1},{b1})"
                color_bot = f"rgb({r2},{g2},{b2})"

                text.append("▀", style=Style(color=color_top, bgcolor=color_bot))
            text.append("\n")

        return text

    def action_open_viewer(self):
        """Open the preview image in external viewer."""
        try:
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", str(self.preview_path)])
            elif sys.platform == "darwin": # macOS
                subprocess.Popen(["open", str(self.preview_path)])
            elif sys.platform == "win32":
                subprocess.Popen(["start", str(self.preview_path)], shell=True)
            self.notify("Opened external viewer")
        except Exception as e:
            self.notify(f"Failed to open viewer: {e}", severity="error")

    def action_save_output(self):
        """Save to final filename and quit."""
        # Use main's naming logic
        from main import get_output_filename
        final_path = get_output_filename(self.image_path)
        shutil.copy(self.preview_path, final_path)
        print(f"Saved to {final_path}")
        self.exit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tui.py <image_path>")
        sys.exit(1)

    app = DitherApp(sys.argv[1])
    app.run()
