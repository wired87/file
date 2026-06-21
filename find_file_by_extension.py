import os
from pathlib import Path
from typing import Iterable, Union

def find_file_by_extension(target_path: Union[str, Path, Iterable], trgt_ftype: str) -> str | None:
    """
    Sucht nach der ersten Datei, die auf `trgt_ftype` endet.
    Akzeptiert entweder einen Ordnerpfad oder eine Liste von Dateien.
    """
    trgt_ftype = trgt_ftype.lower()

    # Falls ein einzelner Pfad/Ordner übergeben wurde, listen wir dessen Inhalt auf
    if isinstance(target_path, (str, Path)):
        path_obj = Path(target_path)
        if path_obj.is_dir():
            # Holt alle Dateien aus dem Ordner (flach, nicht rekursiv)
            files_to_check = [str(f) for f in path_obj.iterdir() if f.is_file()]
        else:
            files_to_check = [str(target_path)]
    else:
        # Falls es bereits eine Liste/Iterable ist
        files_to_check = [str(f) for f in target_path]

    # Die eigentliche Suche
    for file in files_to_check:
        if file.lower().endswith(trgt_ftype):
            return file

    return None