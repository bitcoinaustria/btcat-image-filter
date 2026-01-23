from pathlib import Path
from typing import Optional
from textual.app import App
from .screens import FileSelectionScreen, DitheringScreen

class DitherApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    """

    def __init__(self, initial_image: Optional[str] = None):
        super().__init__()
        self.initial_image = initial_image

    def on_mount(self):
        if self.initial_image:
             self.push_screen(DitheringScreen(Path(self.initial_image)))
        else:
             self.push_screen(FileSelectionScreen())
