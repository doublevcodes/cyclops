from pathlib import Path
import pickle

import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")  # Pickle files are serialized Python objects

Path("train").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("val").mkdir(exist_ok=True)

def encode_known_faces(
        model: str = "hog",  # HOG (Histogram of Oriented Gradients) is a preferred model for CPU-based hardware
                             # CNN (Convolutional Neural Network) is used on GPU-accelereted hardware
        encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("train").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
    
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()