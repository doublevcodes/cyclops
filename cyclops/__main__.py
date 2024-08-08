from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import logging

from rich.logging import RichHandler
import typer

from cyclops import ModelChoice
from cyclops.recognise import recognise_faces
from cyclops.train import encode


logging.basicConfig(handlers=[RichHandler()])


def debug_callback():
    logging.basicConfig(level=logging.DEBUG)


cyclops = typer.Typer(
    no_args_is_help=True,
)


@cyclops.command()
def train(
    path: Annotated[Path, typer.Argument(exists=True)],
    model: Optional[Annotated[ModelChoice, typer.Option()]] = ModelChoice.HOG,
    debug: Optional[Annotated[bool, typer.Option(callback=debug_callback)]] = False,
):
    encode(path, model)


@cyclops.command()
def recognise(
    path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Optional[Annotated[ModelChoice, typer.Option()]] = ModelChoice.HOG,
    debug: Optional[Annotated[bool, typer.Option(callback=debug_callback)]] = False,
):
    recognise_faces(path, model)


if __name__ == "__main__":
    cyclops()
