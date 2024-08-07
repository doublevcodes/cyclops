from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer

from cyclops import ModelChoice
from cyclops.recognise import recognise_faces
from cyclops.train import encode


APP_NAME = "cyclops"

cyclops = typer.Typer(
    no_args_is_help=True,
)

@cyclops.command()
def train(
    path: Annotated[Path, typer.Argument(exists=True)],
    model: Optional[Annotated[ModelChoice, typer.Option()]] = ModelChoice.HOG
):
    encode(path, model)

@cyclops.command()
def recognise(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Optional[Annotated[ModelChoice, typer.Option()]] = ModelChoice.HOG
):
    recognise_faces(path, model)

@cyclops.command()
def who():
    print("Me")

if __name__ == "__main__":
    cyclops()