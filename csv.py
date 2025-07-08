import csv
from io import StringIO, BytesIO

from utils.printer import printer


def dict_2_csv(data:list[dict], keys):
    print("Save local scv data")
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def dict_2_csv_buffer(data: list[dict], keys, bbuffer, testing=True, save_dir=None) -> BytesIO:
    """
    Schreibt die CSV in einen BytesIO-Puffer und gibt diesen zurück.
    """
    # StringIO (Text) als Zwischenpuffer
    #printer(locals())
    clean_data = []
    if testing is True:
        with open(save_dir, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    else:
        str_buf = StringIO()
        writer = csv.DictWriter(str_buf, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
        byte_buf = BytesIO()
        # In BytesIO konvertieren (UTF-8 Encoding)
        byte_buf.write(str_buf.getvalue().encode("utf-8"))
        return byte_buf
