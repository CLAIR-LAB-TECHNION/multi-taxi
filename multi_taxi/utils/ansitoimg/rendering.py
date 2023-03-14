"""Render the ANSI

adapted from:
https://github.com/FHPythonUtils/AnsiToImg
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ansirep import AnsiBlocks, findLen
from .theme import THEME

THISDIR = str(Path(__file__).resolve().parent)

# monospaced chars have a constant height and width
TEXT_HEIGHT = 15
TEXT_WIDTH = 8.7

# Load the fonts
FONT_NORMAL = ImageFont.truetype(f"{THISDIR}/resources/FiraCode-Regular.otf", 14)
FONT_BOLD = ImageFont.truetype(f"{THISDIR}/resources/FiraCode-Bold.otf", 14)
FONT_ITALIC = ImageFont.truetype(f"{THISDIR}/resources/FiraCode-Italic.otf", 14)
FONT_BOLD_ITALIC = ImageFont.truetype(f"{THISDIR}/resources/FiraCode-BoldItalic.otf", 14)
FONT_EMOJI = ImageFont.truetype(f"{THISDIR}/resources/TwitterColorEmoji-SVGinOT30.ttf", 14)


class TaxiMapRendering:
    def __init__(self, map_arr):
        self.__initial_map_arr = map_arr.copy()
        self.__width = len(map_arr[0]) + 1
        map_ansi = '\n'.join(''.join(row) for row in map_arr)

        self.themeData = THEME
        ansiBlocks = AnsiBlocks(map_ansi)
        ansiBlocks.process()
        blocks = ansiBlocks.ansiBlocks
        size = (int(self.__width * TEXT_WIDTH), int(TEXT_HEIGHT * ansiBlocks.height + 5))
        self.image = Image.new("RGB", size, "#" + self.themeData["base00"])
        self.__initial_map_render(blocks)

        self.cur_img = self.image

    def draw_taxis_and_passengers(self, map_with_entities):
        map_ansi = '\n'.join(''.join(row) for row in map_with_entities)

        ansiBlocks = AnsiBlocks(map_ansi)
        ansiBlocks.process()
        blocks = ansiBlocks.ansiBlocks

        self.cur_img = self.image.copy()

        draw = ImageDraw.Draw(self.cur_img)

        # Iterate through the ANSI blocks
        for block in blocks:
            # skip blocks already in the original map
            initial_value_at_arr = self.__initial_map_arr[block.position[1], block.position[0]]
            value_at_arr = map_with_entities[block.position[1], block.position[0]]
            if initial_value_at_arr == value_at_arr:
                continue

            self.__render_block(block, draw)

    def get_cur_image_crop(self, bottom, top, right, left):
        # fit crop to image
        bottom = int(bottom * TEXT_HEIGHT) + 3
        top = int(top * TEXT_HEIGHT) + 4
        right = int(right * TEXT_WIDTH) - 5
        left = int(left * TEXT_WIDTH) + 4

        # find out correct paddings according to crop
        w, h = self.cur_img.size
        bottom_pad = max(0, bottom - h)
        top_pad = max(0, abs(top))
        right_pad = max(0, right - w)
        left_pad = max(0, abs(left))

        if any(pad > 0 for pad in [bottom_pad, top_pad, right_pad, left_pad]):
            padded_img = self.__add_margin(self.cur_img, top_pad, right_pad, bottom_pad, left_pad, (40, 44, 52))
        else:
            padded_img = self.cur_img

        bottom += top_pad
        top += top_pad
        right += left_pad
        left += left_pad

        return padded_img.crop((left, top, right, bottom))

    @staticmethod
    def __add_margin(pil_img, top, right, bottom, left, color):
        # solution from:
        # https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def __initial_map_render(self, blocks):
        draw = ImageDraw.Draw(self.image)

        # Iterate through the ANSI blocks
        for block in blocks:
            self.__render_block(block, draw)

    def __render_block(self, block, draw):
        posY = block.position[1] * TEXT_HEIGHT + 2.5
        if block.bgColour is not None:
            posX = block.position[0] * TEXT_WIDTH + 5
            draw.rectangle(
                (posX, posY, posX + findLen(block.text) * TEXT_WIDTH, posY + TEXT_HEIGHT),
                block.bgColour,
            )
        text = block.text
        font = FONT_NORMAL
        fill = (
            "#" + self.themeData["base05"] if block.fgColour is None else block.fgColour
        )  # get the block fill colour
        if block.bold and block.italic:
            font = FONT_BOLD_ITALIC
        elif block.bold:
            font = FONT_BOLD
        elif block.italic:
            font = FONT_ITALIC
        index = 0
        for char in text:
            posX = (block.position[0] + index) * TEXT_WIDTH + 5
            if ord(char) > 10000:  # I wish there was a better way of doing this...
                draw.text((posX, posY + 2), char, font=FONT_EMOJI, fill=fill)
                index += 2
            else:
                draw.text(
                    (posX, posY),
                    char if not block.crossedOut else "\u0336{char}\u0336",
                    font=font,
                    fill=fill,
                )
                index += 1
            if block.underline:
                draw.line(
                    (posX, posY + TEXT_HEIGHT, posX + 9.5, posY + TEXT_HEIGHT), fill=fill, width=1
                )
