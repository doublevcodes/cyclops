import logging
from pathlib import Path
import pickle

import face_recognition
from rich.progress import track

from cyclops import DEFAULT_ENCODINGS_PATH, ModelChoice


logger = logging.getLogger(__name__)

def encode(
    training_dir: Path,
    model: ModelChoice,
    debug: bool,
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:

    if debug:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"Beginning to encode data from training set in {training_dir.absolute().as_posix()}")

    names = encodings = []

    for folder in [f for f in training_dir.iterdir() if f.is_dir()]:
        logger.info(f"Training on {folder.name}")
        
        for image_file in track([*folder.iterdir()], description="Training..."):
            
            logger.debug(f"Encoding {image_file.name}")

            image = face_recognition.load_image_file(image_file)

            bounding_boxes = face_recognition.face_locations(image, model=model)
            if len(bounding_boxes) != 1:
                continue

            bounding_box = bounding_boxes[0]
            
            face_encodings = face_recognition.face_encodings(image, [bounding_box])
            if (
                len(face_encodings) == 0
            ):
                continue

            face_encoding = face_encodings[0]

            names.append(folder.name)
            encodings.append(face_encoding)

            logger.debug(f"Successfully encoded {image_file.parent}/{image_file.name}")

    name_encodings = {"names": names, "encodings": encodings}

    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
