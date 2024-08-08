from collections import Counter
import logging
from pathlib import Path
import pickle
from PIL import Image, ImageDraw

import face_recognition

from cyclops import DEFAULT_ENCODINGS_PATH, BOUNDING_BOX_COLOR, ModelChoice
from cyclops.media.img import overlay


logger = logging.getLogger(__name__)

def _display_face(draw: ImageDraw.Draw, bounding_box, name):
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


def _recognise_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )

    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )

    logger.debug(f"The following votes were made on faces in the photo: {votes}")

    if votes:
        return votes.most_common(1)[0][0]


def recognise_faces(
    image_location: Path,
    model: ModelChoice = ModelChoice.HOG,
    text: bool = False,
    debug: bool = False,
) -> None:
    
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Attempting to recognise face in image {image_location.absolute().as_posix()}")

    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    logger.debug(f"Image file successfully loaded")

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    logger.debug(f"Image file face locations found and faces successfully encoded")

    pillow_image = Image.fromarray(input_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognise_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        if text:
            logger.debug("Text result was chosen instead of image overlay")
            logger.info(f"{name} found in bounding box: {bounding_box}")
        else:
            logger.debug(f"{name} found in bounding box: {bounding_box}")
            overlay(pillow_image, bounding_box, name)

    if not text:
        pillow_image.show()
