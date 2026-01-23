#!/usr/bin/env python3
# Copyright 2026 Harald Schilly <hsy@bitcoin-austria.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image dithering tool that applies monochrome dithering to a portion of an image.
Uses Austrian flag red (#E3000F) and applies dithering to the right side after a golden ratio cut.
"""

from btcat_images.cli import main
from btcat_images.core.pipeline import apply_dither, dither_image
from btcat_images.core.utils import get_output_filename
from btcat_images.constants import GOLDEN_RATIO, BRANDS, DitherPattern, DARK_BACKGROUND
from btcat_images.processing.filters.glitch import glitch_swap_rows
from btcat_images.processing.filters.glitch import glitch_swap_rows

if __name__ == '__main__':
    main()