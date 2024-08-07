from pathlib import Path
import pickle

import face_recognition

from cyclops import DEFAULT_ENCODINGS_PATH, ModelChoice

def encode(
        training_dir: Path,
        model: ModelChoice,
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    
    names = []
    encodings = []
    
    for filepath in training_dir.glob("*/*"):
        print(filepath.as_posix())
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)

        if len(face_locations) > 1:
            # TODO: implement an error because there's more than one face
            continue

        face_encodings = face_recognition.face_encodings(image, face_locations)
        if len(face_encodings) == 0:  # A face was unable to be recognised in the training image
            continue

        face_encoding = face_encodings[0]

        names.append(name)
        encodings.append(face_encoding)

    name_encodings = {"names": names, "encodings": encodings}
    
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)