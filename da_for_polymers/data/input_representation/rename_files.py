from pathlib import Path
import os


def rename_files(path: Path):
    """_summary_

    Args:
        path (Path): _description_
    """
    for p in path.rglob("*valid*"):
        renamed: str = p.name
        renamed: str = renamed.replace("valid", "test")
        p.rename(p.parent / renamed)


if __name__ == "__main__":
    path: Path = Path(os.getcwd())
    rename_files(path)
