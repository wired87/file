import csv


def dict_2_csv(data:list[dict], keys):
    print("Save local scv data")
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)