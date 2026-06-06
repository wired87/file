from pathlib import Path
import pyarrow.parquet as pq


class ParquetMaster:
    def __init__(self, path: str):
        self.path = Path(path)
        self.parquet = pq.ParquetFile(self.path)

    def read(
            self,
            return_dict: bool = True,
            print_specs: bool = True,
    ):
        metadata = self.parquet.metadata

        specs = {
            "file": str(self.path),
            "size_bytes": self.path.stat().st_size,
            "size_mb": round(self.path.stat().st_size / 1024 / 1024, 2),
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "num_columns": metadata.num_columns,
            "columns": self.parquet.schema.names,
            "schema": str(self.parquet.schema),
        }

        if print_specs:
            print("\n=== PARQUET FILE ===")
            print(f"file           : {specs['file']}")
            print(f"size_mb        : {specs['size_mb']}")
            print(f"rows           : {specs['num_rows']}")
            print(f"row_groups     : {specs['num_row_groups']}")
            print(f"columns        : {specs['num_columns']}")

            print("\n=== COLUMN NAMES ===")
            for col in specs["columns"]:
                print(col)

            print("\n=== SCHEMA ===")
            print(specs["schema"])

        if return_dict:
            return specs

        return None

    def iter_batches(
            self,
            batch_size: int = 10000,
            columns=None
    ):
        for batch in self.parquet.iter_batches(
                batch_size=batch_size,
                columns=columns
        ):
            yield batch