from pathlib import Path
import pickle

import face_recognition

from cyclops import DEFAULT_ENCODINGS_PATH

def encode(
        training_dir,
        model,
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    
    names = []
    encodings = []
    
    for filepath in training_dir.glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if len(face_encodings) > 1:
            # Error due to multiple faces in the image.
            pass

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        encodings = {"names": names, "encodings": encodings}
    
    with encodings_location.open(mode="wb") as f:
        pickle.dump(encodings, f)