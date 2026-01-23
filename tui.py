#!/usr/bin/env python3
import sys
from btcat_images.tui.app import DitherApp

if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Requirement: clearly mention to start it with uv run python
    if image_arg is None:
        print("Note: Run with 'uv run python tui.py [image_path]'")
        print("Starting file selection...")

    app = DitherApp(image_arg)
    app.run()