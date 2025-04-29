import yaml

def load_yaml(filepath) -> dict:
    with open(filepath, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    #print("data", data)
    return data


def write_yaml(content:dict, dest):
    yml_content = yaml.dump(content, default_flow_style=False, sort_keys=False)
    with open(dest, 'r', encoding="utf-8") as f:
        f.write(yml_content)
