from PIL import Image, ImageDraw

from cyclops import BOUNDING_BOX_COLOR

def overlay(image: Image.Image, bounding_box: tuple[int, int, int, int], name: str):

    draw = ImageDraw.Draw(image)

    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

    del draw