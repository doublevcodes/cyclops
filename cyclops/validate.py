from pathlib import Path

from cyclops.recognise import recognise_faces


def validate(model: str = "hog"):
    for filepath in Path("val").rglob("*"):
        if filepath.is_file():
            recognise_faces(image_location=str(filepath.absolute()), model=model)
