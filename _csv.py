import csv
from io import StringIO

def dict_2_csv(data:list[dict], keys):
    print("Save local scv data")
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def dict_2_csv_buffer(data: list[dict], keys, save_dir) -> None:
    """
    Schreibt die CSV in einen BytesIO-Puffer und gibt diesen zurück.
    """
    # StringIO (Text) als Zwischenpuffer
    #printer(locals())
    str_buf = StringIO()
    writer = csv.DictWriter(str_buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(data)

    # Write csv directly to save (local, or tmp)
    with open(save_dir, "w", encoding="utf-8") as f:
        f.write(str_buf.getvalue())
