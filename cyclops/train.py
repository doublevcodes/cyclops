import logging
from pathlib import Path
import pickle

import face_recognition
from rich.progress import track

from cyclops import DEFAULT_ENCODINGS_PATH, ModelChoice


logger = logging.getLogger(__name__)

def encode(
    training_dir: Path,
    model: ModelChoice = ModelChoice.HOG,
    debug = False,
) -> None:
    
    if debug:
        logger.setLevel(logging.DEBUG)
    
    names = []
    encodings = []

    for folder in [f for f in training_dir.iterdir() if f.is_dir()]:

        logger.info(f"Encoding images in {folder.name}")

        for image_file in track([*folder.iterdir()], description="Training..."):

            logger.debug(f"Encoding {folder.name}/{image_file.name}")

            name = image_file.parent.name
            image = face_recognition.load_image_file(image_file)

            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}

    with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
