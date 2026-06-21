import os
from pathlib import Path
from typing import Iterable


def save_files(
        files: Iterable,
        output_dir: str | Path,
        outsrc_ftype=None,
        include_file_type=None
) -> list[Path]:
    """
    Receives Django uploaded files and writes them into output_dir based on filters.

    Args:
        files: List of Django UploadedFile objects (e.g. TemporaryUploadedFile)
        output_dir: Target directory

    Returns:
        List of written file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []

    for file in files:
        # 1. Access the filename via the .name attribute
        filename = file.name

        # Skip if outsrc_ftype is provided and matches
        if outsrc_ftype and filename.endswith(outsrc_ftype):
            continue

        # Skip if include_file_type is provided and does NOT match
        if include_file_type and not filename.endswith(include_file_type):
            continue

        target_path = output_dir / os.path.basename(filename)

        # 2. Safely stream chunked data out to the temporary directory
        with open(target_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        written_paths.append(target_path)

    return written_paths