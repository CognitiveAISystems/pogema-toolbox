import yaml


def maps_dict_to_yaml(filename, maps):
    with open(filename, 'w') as file:
        yaml.add_representer(str,
                             lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|'))
        yaml.dump(maps, file)
