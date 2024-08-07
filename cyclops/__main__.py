import typer

cyclops = typer.Typer()

@cyclops.command()
def main():
    print("Hello Typer")

@cyclops.command()
def who():
    print("Me")

if __name__ == "__main__":
    cyclops()