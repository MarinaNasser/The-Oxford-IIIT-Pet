import os

def load_image_class_mapping(list_file, images_folder):
    image_class_mapping = {}
    with open(list_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            image_name = parts[0]
            class_id = int(parts[1])
            image_path = os.path.join(images_folder, image_name + '.jpg')
            if os.path.exists(image_path):
                image_class_mapping[image_path] = class_id
    return image_class_mapping
