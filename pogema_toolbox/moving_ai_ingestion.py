import requests
import zipfile
import io

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


def download_moving_ai_maps(url):
    response = requests.get(url)

    zip_file = io.BytesIO(response.content)

    z = zipfile.ZipFile(zip_file, 'r')

    maps_dict = {}

    for file_name in z.namelist():
        if file_name.endswith('.map'):
            with z.open(file_name) as f:
                grid = map_to_grid(f)
                maps_dict[file_name.replace('.map', "")] = grid

    z.close()

    return maps_dict


def map_to_grid(file_in_zip, remove_border=True):
    lines = []
    with file_in_zip as f:
        type_ = f.readline().decode('utf-8').split(' ')[1]
        height = int(f.readline().decode('utf-8').split(' ')[1])
        width = int(f.readline().decode('utf-8').split(' ')[1])
        _ = f.readline()

        for _ in range(height):
            line = f.readline().decode('utf-8').rstrip()
            lines.append(line)

    m = []
    rmb = 1 if remove_border else 0
    for i in range(rmb, len(lines) - rmb):
        line = []
        for j in range(rmb, len(lines[i]) - rmb):
            symbol = lines[i][j]
            is_obstacle = symbol in ['@', 'O', 'T']
            line.append('#' if is_obstacle else '.')
        m.append("".join(line))
    return '\n'.join(m)


def main():
    url = 'https://movingai.com/benchmarks/street/street-map.zip'
    maps = download_moving_ai_maps(url)
    maps_dict_to_yaml('maps.yaml', maps)


if __name__ == '__main__':
    main()
