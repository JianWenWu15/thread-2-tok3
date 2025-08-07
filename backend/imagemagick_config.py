# ImageMagick configuration for MoviePy
import os
import moviepy.config as config

# Set the path to ImageMagick
IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

# Configure MoviePy to use ImageMagick
if os.path.exists(IMAGEMAGICK_BINARY):
    config.IMAGEMAGICK_BINARY = IMAGEMAGICK_BINARY
    print(f"ImageMagick configured at: {IMAGEMAGICK_BINARY}")
else:
    print("ImageMagick not found at expected location")
