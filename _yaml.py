import yaml

def load_yaml(filepath) -> dict:
    with open(filepath, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data



def write_yaml(content: dict | str, dest):
    print("Write yaml content to dest", dest)

    # Falls der Content bereits ein fertiger String ist, nutzen wir ihn direkt
    if isinstance(content, str):
        yml_content = content
    else:
        # Falls es ein Dictionary ist, serialisieren wir es wie gewohnt
        yml_content = yaml.dump(content, default_flow_style=False, sort_keys=False)

    with open(dest, 'w', encoding="utf-8") as f:
        f.write(yml_content)
