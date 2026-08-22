"""High-Quality Motion Graphics Card Generator using Pillow.

Generates broadcast-grade typography cards, stat cards, and quote cards
with radial glow, grid patterns, glassmorphism containers, and word wrapping.
"""

import os
import textwrap
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Attempts to load high-quality system fonts, with fallback."""
    font_candidates = [
        # macOS Fonts
        "/System/Library/Fonts/SFPro-Bold.ttf" if bold else "/System/Library/Fonts/SFPro-Regular.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in font_candidates:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def generate_motion_card_image(
    output_path: str,
    hero_text: str,
    label: str = "STATISTICAL HIGHLIGHT",
    subtext: str = "",
    theme_color: str = "#3B82F6",
    card_type: str = "stat",
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Generates a pristine 1080p/4K motion graphics card with glassmorphism and glowing effects."""
    # 1. Base dark canvas
    img = Image.new("RGBA", (width, height), (6, 9, 19, 255))
    draw = ImageDraw.Draw(img)

    # 2. Cyber grid backdrop
    grid_size = int(width / 28)
    grid_color = (59, 130, 246, 22)
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # 3. Dynamic radial glow orbs
    glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    cx, cy = width // 2, height // 2
    r1 = int(min(width, height) * 0.38)
    glow_draw.ellipse(
        [cx - r1, cy - r1, cx + r1, cy + r1],
        fill=(59, 130, 246, 75) if theme_color == "#3B82F6" else (147, 51, 234, 65),
    )

    r2 = int(min(width, height) * 0.22)
    glow_draw.ellipse(
        [cx - int(r2 * 0.8), cy - int(r2 * 0.8), cx + int(r2 * 0.8), cy + int(r2 * 0.8)],
        fill=(6, 182, 212, 55),
    )
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=int(min(width, height) * 0.14)))
    img = Image.alpha_composite(img, glow_layer)
    draw = ImageDraw.Draw(img)

    # 4. Glassmorphic Hero Card Container
    card_w = int(width * 0.72)
    card_h = int(height * 0.62)
    card_x0 = (width - card_w) // 2
    card_y0 = (height - card_h) // 2
    card_x1 = card_x0 + card_w
    card_y1 = card_y0 + card_h
    radius = int(min(width, height) * 0.03)

    # Shadow layer
    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.rounded_rectangle(
        [card_x0 + 4, card_y0 + 8, card_x1 + 4, card_y1 + 8],
        radius=radius,
        fill=(0, 0, 0, 150),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=30))
    img = Image.alpha_composite(img, shadow_layer)
    draw = ImageDraw.Draw(img)

    # Card background & border
    card_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card_layer)
    card_draw.rounded_rectangle(
        [card_x0, card_y0, card_x1, card_y1],
        radius=radius,
        fill=(15, 23, 42, 230),
        outline=(59, 130, 246, 120) if theme_color == "#3B82F6" else (147, 51, 234, 120),
        width=2,
    )
    img = Image.alpha_composite(img, card_layer)
    draw = ImageDraw.Draw(img)

    # 5. Top Pill Tag Badge (e.g. "• APOLLO PROJECT INVESTMENT")
    badge_font = _get_font(int(height * 0.024), bold=True)
    badge_text = f"• {label.upper()}"
    badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_tw = badge_bbox[2] - badge_bbox[0]
    badge_th = badge_bbox[3] - badge_bbox[1]

    badge_pad_x = int(width * 0.015)
    badge_pad_y = int(height * 0.008)
    badge_bx0 = cx - (badge_tw // 2) - badge_pad_x
    badge_by0 = card_y0 + int(card_h * 0.11)
    badge_bx1 = cx + (badge_tw // 2) + badge_pad_x
    badge_by1 = badge_by0 + badge_th + (badge_pad_y * 2)

    draw.rounded_rectangle(
        [badge_bx0, badge_by0, badge_bx1, badge_by1],
        radius=int(badge_th),
        fill=(23, 37, 84, 235),
        outline=(59, 130, 246, 170),
        width=1,
    )
    draw.text(
        (cx - (badge_tw // 2), badge_by0 + badge_pad_y),
        badge_text,
        fill=(147, 197, 253, 255),
        font=badge_font,
    )

    # 6. Hero Metric / Large Quote Text
    hero_font_size = int(height * 0.115) if card_type == "stat" else int(height * 0.055)
    hero_font = _get_font(hero_font_size, bold=True)

    if card_type == "stat":
        hero_display = hero_text.strip()
        hero_bbox = draw.textbbox((0, 0), hero_display, font=hero_font)
        hero_tw = hero_bbox[2] - hero_bbox[0]
        hero_th = hero_bbox[3] - hero_bbox[1]
        hero_y = badge_by1 + int(card_h * 0.06)

        # Drop shadow on hero text
        draw.text((cx - (hero_tw // 2) + 2, hero_y + 3), hero_display, fill=(2, 132, 199, 160), font=hero_font)
        draw.text((cx - (hero_tw // 2), hero_y), hero_display, fill=(56, 189, 248, 255), font=hero_font)
        next_y = hero_y + hero_th + int(card_h * 0.06)
    else:
        hero_lines = textwrap.wrap(f"“{hero_text}”", width=38)
        hero_y = badge_by1 + int(card_h * 0.06)
        line_spacing = int(hero_font_size * 1.3)
        for idx, line in enumerate(hero_lines):
            l_bbox = draw.textbbox((0, 0), line, font=hero_font)
            l_tw = l_bbox[2] - l_bbox[0]
            draw.text((cx - (l_tw // 2), hero_y + (idx * line_spacing)), line, fill=(248, 250, 252, 255), font=hero_font)
        next_y = hero_y + (len(hero_lines) * line_spacing) + int(card_h * 0.04)

    # 7. Subtext with clean multi-line word wrapping
    if subtext:
        sub_font_size = int(height * 0.032)
        sub_font = _get_font(sub_font_size, bold=False)
        wrap_width = 52 if width >= 1920 else 40
        wrapped_lines = textwrap.wrap(subtext, width=wrap_width)
        sub_line_spacing = int(sub_font_size * 1.45)

        for idx, line in enumerate(wrapped_lines[:4]):
            line_bbox = draw.textbbox((0, 0), line, font=sub_font)
            line_tw = line_bbox[2] - line_bbox[0]
            draw.text(
                (cx - (line_tw // 2), next_y + (idx * sub_line_spacing)),
                line,
                fill=(203, 213, 225, 255),
                font=sub_font,
            )

    # 8. Decorative indicator bottom accents
    accent_y = card_y1 - int(card_h * 0.1)
    bar_w = int(width * 0.04)
    draw.rounded_rectangle([cx - bar_w - 15, accent_y, cx - 15, accent_y + 3], radius=2, fill=(59, 130, 246, 220))
    draw.ellipse([cx - 4, accent_y - 2, cx + 4, accent_y + 6], fill=(6, 182, 212, 255))
    draw.rounded_rectangle([cx + 15, accent_y, cx + bar_w + 15, accent_y + 3], radius=2, fill=(59, 130, 246, 220))

    # Save to RGB PNG
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.convert("RGB").save(output_path, "PNG", quality=95)
    return output_path
