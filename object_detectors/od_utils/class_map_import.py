import json

def import_json(json_path,swap_kvp=False,training_script_init=None):
    with open(json_path, 'r') as file:
        class_map = json.load(file)
    if swap_kvp:
        swapped_class_map = {value: key for key, value in class_map.items()}
        class_map=swapped_class_map
    if training_script_init is not None:
        header="path: /app/data # dataset root dir (must use absolute path!)\ntrain: images  # train images (relative to 'path')\nval: images  # val images (relative to 'path')\ntest:  # test images (optional)\n\nnames:  # class names must match with the names in class_map.json\n"
        with open(training_script_init,'w') as file:
            file.write(header)
            for key in class_map.keys():
                line=f'  {class_map[key]}: {key}\n'
                file.write(line)
    return class_map

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--json_path', required=True, help='the path to images')
    ap.add_argument('--swap_key_value_pairs', action='store_true', help='Swap key value pairs.')
    ap.add_argument('--training_script_path', default=None, help='Initial training script path.')
    args = vars(ap.parse_args())
    import_json(args['json_path'],args['swap_key_value_pairs'],args['training_script_path'])