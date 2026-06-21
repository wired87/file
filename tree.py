from pathlib import Path


def debug_print_tree(directory_path: str | Path, prefix: str = ""):
    """
    Recursively walks through a directory and logs a visual tree structure
    of all nested files, subdirectories, and their respective sizes.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        print(f"Directory does not exist: {dir_path}")
        return

    # Print root entry if it's the top-level call
    if prefix == "":
        print(f"\n📁 [ROOT] {dir_path.resolve()}")

    try:
        entries = sorted(list(dir_path.iterdir()), key=lambda e: (e.is_file(), e.name.lower()))
    except Exception as e:
        print(f"{prefix}└── [ERROR accessing directory: {e}]")
        return

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "

        if entry.is_dir():
            print(f"{prefix}{connector}📂 {entry.name}/")
            # Recurse down into nested tree structure
            next_prefix = prefix + ("    " if is_last else "│   ")
            debug_print_tree(entry, next_prefix)
        else:
            # Calculate human-readable file size
            try:
                size_kb = entry.stat().st_size / 1024
                size_str = f"({size_kb:.1f} KB)"
            except Exception:
                size_str = "(unknown size)"

            print(f"{prefix}{connector}📄 {entry.name} {size_str}")