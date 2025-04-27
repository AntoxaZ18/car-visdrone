
def get_yaml_config(yaml_cfg :str, section: str):
    import yaml
    with open(yaml_cfg) as stream:
        try:
            return yaml.safe_load(stream)[section]
        except yaml.YAMLError as exc:
            print(exc)
