#!/usr/bin/env python3
"""Generate YouTube channel art from the existing Fourth & Value brand.

Colours and the wordmark come from docs/assets/brand/logo-monogram.svg and
docs/assets/logo-fv.svg so the channel matches the site rather than
introducing a second visual identity.

YouTube crops the banner differently on every device. Only a 1235x338 box in
the centre is guaranteed visible, so all text is kept inside it and the rest
of the 2048x1152 canvas is background that can be safely cut.

  python scripts/make_channel_art.py --out-dir docs/assets/brand
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

# Site palette: docs/assets/site.css and the brand SVGs.
BG = (11, 14, 19)          # --bg #0b0e13
INK = (231, 231, 239)      # wordmark "Fourth"
BLUE = (76, 111, 255)      # the ampersand
GREEN = (34, 197, 94)      # "Value" and the monogram field
DARK = (15, 20, 24)        # monogram letterforms
MUTED = (139, 147, 161)

BANNER = (2048, 1152)
SAFE = (1235, 338)         # guaranteed-visible centre box
AVATAR = 800

FONT_CANDIDATES = [
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
]


def load_font(size, bold=True):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_width(draw, text, font):
    return draw.textbbox((0, 0), text, font=font)[2]


def draw_wordmark(draw, cx, cy, size):
    """"Fourth & Value" centred on (cx, cy), coloured like the brand SVG."""
    font = load_font(size)
    parts = [('Fourth ', INK), ('& ', BLUE), ('Value', GREEN)]
    total = sum(text_width(draw, t, font) for t, _ in parts)
    x = cx - total / 2
    for text, colour in parts:
        draw.text((x, cy), text, font=font, fill=colour, anchor='lm')
        x += text_width(draw, text, font)
    return total


def make_banner(path):
    img = Image.new('RGB', BANNER, BG)
    cx, cy = BANNER[0] // 2, BANNER[1] // 2

    # Symmetric vertical glow centred on the safe area. Fading in both
    # directions avoids the hard seam a one-way ramp leaves behind.
    glow = Image.new('RGB', BANNER, BG)
    gd = ImageDraw.Draw(glow)
    reach = 330
    for dy in range(-reach, reach):
        t = 1 - abs(dy) / reach
        fade = t * t                      # ease so the edges reach BG smoothly
        gd.line([(0, cy + dy), (BANNER[0], cy + dy)],
                fill=(int(BG[0] + 10 * fade),
                      int(BG[1] + 30 * fade),
                      int(BG[2] + 18 * fade)))
    img = Image.blend(img, glow, 1.0)
    draw = ImageDraw.Draw(img)

    # Block is centred as a whole rather than hung off the wordmark baseline.
    draw_wordmark(draw, cx, cy - 52, 132)

    tag = 'We watch the numbers so you can watch the game.'
    draw.text((cx, cy + 48), tag, font=load_font(46), fill=MUTED, anchor='mm')

    sub = 'NFL betting insights  ·  line shopping  ·  fourthandvalue.com'
    draw.text((cx, cy + 116), sub, font=load_font(34), fill=GREEN, anchor='mm')

    x0, x1 = cx - SAFE[0] // 2, cx + SAFE[0] // 2
    draw.line([(x0, cy + 152), (x1, cy + 152)], fill=(40, 52, 68), width=2)

    img.save(path, 'PNG', optimize=True)
    return path


def make_avatar(path):
    """Round-safe: YouTube masks the picture to a circle."""
    size = AVATAR
    img = Image.new('RGB', (size, size), GREEN)
    draw = ImageDraw.Draw(img)

    # Sized to sit comfortably inside YouTube's circular mask.
    font = load_font(int(size * 0.38))
    draw.text((size / 2, size / 2), 'FV', font=font, fill=DARK, anchor='mm')

    img.save(path, 'PNG', optimize=True)
    return path


def main():
    ap = argparse.ArgumentParser(description='Generate YouTube channel art')
    ap.add_argument('--out-dir', default='docs/assets/brand')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    b = make_banner(os.path.join(args.out_dir, 'youtube-banner.png'))
    a = make_avatar(os.path.join(args.out_dir, 'youtube-avatar.png'))
    for p in (b, a):
        with Image.open(p) as im:
            print(f'{p}  {im.size[0]}x{im.size[1]}  {os.path.getsize(p)/1024:.0f} KB')


if __name__ == '__main__':
    main()
