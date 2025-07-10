import os
import zipfile
from io import BytesIO

from file.temp import rm_tmp


def zip_tempdir(tempdir_path, zip_dest: BytesIO or str):
    with zipfile.ZipFile(zip_dest, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(tempdir_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, tempdir_path)  # ZIP: relative Pfade
                zf.write(abs_path, arcname=rel_path)  # Inhalt + Pfadstruktur erhalten
                print(f"Transfered file {file} to zip-buffer")

    if isinstance(zip_dest, BytesIO):
        size = len(zip_dest.getvalue())
        print(f"Buffer-size: {size} Bytes")
        print("Mark buffer as OK")

    rm_tmp(tempdir_path)