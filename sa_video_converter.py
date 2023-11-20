import json
import os.path
import random
import cv2


"""
This script is used to auto generate training dataset in yolov5 format from the 
downloaded folder of superannotate video project.

The SA folder structure should look like this

--  classes\classes.json
--  video1.json
--  video2.json
--  video3.json
......

"""


def generate_yolo(sa_folder_path, output_name="yolo_data", capture_rate: int = 30, split: dict = {}):
    """
    main function: create class name using classes/class.json and process all json file in the project folder
    Args:
        sa_folder_path: folder path
        output_name: output folder name
        capture_rate: in second
        split: split dictionary
    Returns:
        null
    """
    global CLASSES_DICT
    CLASSES_DICT = convert_class_id(sa_folder_path)
    json_files = [f for f in os.listdir(sa_folder_path) if f.endswith('.json')]
    _valid_dir(output_name, split)
    for index, json_file in enumerate(json_files):
        print(f"Processing video {index + 1}/{len(json_files)}...")
        convert_from_json(json_file, output_name, capture_rate=capture_rate, split=split)
    if split:
        print("Creating data.yaml file")
        create_data_yaml(CLASSES_DICT, output_name)


def convert_class_id(folder_path):
    """
    create class id, starting from 0, 1, 2 ....
    Args:
        folder_path: sa folder path
    Returns:
        class dictionary
    """
    classes_rel_path = r"classes\classes.json"
    classes_abs_path = os.path.join(folder_path, classes_rel_path)
    with open(classes_abs_path) as f:
        classes = json.load(f)
    return {item["name"]: index for index, item in enumerate(classes)}


def _valid_dir(output_name="yolo_data", split: dict = {}):
    """
    helper function: create folders
    Args:
        output_name: output folder name
        split: split dictionary
    Returns:
        null
    """
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    else:
        print("Folder exists...will overwrite to the current folder")
    if split:
        for name in split.keys():
            for i in ["labels", "images"]:
                if not os.path.exists(os.path.join(output_name, f'{name}/{i}')):
                    os.makedirs(os.path.join(output_name, f'{name}/{i}'))


def create_data_yaml(CLASSES_DICT, output_name):
    """
    create data yaml
    Args:
        CLASSES_DICT:
        output_name: output folder name
    Returns:
        null
    """
    with open(f"{output_name}/data.yaml", 'w') as file:
        file.write("names:\n")
        for name in CLASSES_DICT.keys():
            file.write(f'- {name}\n')
        file.write(f'nc: {len(CLASSES_DICT)}\n')
        file.write(f"test: {output_name}/test/images\n")
        file.write(f"train: {output_name}/train/images\n")
        file.write(f"val: {output_name}/val/images\n")


def split_dataset(split):
    """
    generate a random number to decide where this image belongs to
    Args:
        split: split_dict
    Returns:
        "train", "val" or "test"
    """
    total_percentage = sum(split.values())
    if total_percentage != 100:
        raise ValueError("The sum of split must be 100.")
    rand_num = random.randint(1, total_percentage)
    current_percentage = 0
    for key, percentage in split.items():
        current_percentage += percentage
        if rand_num <= current_percentage:
            return key


def convert_from_json(json_filename, output_name, capture_rate: int = 30, split: dict = {}):
    """
    convert single json file to yolov5 txt format
    Args:
        json_filename: sa annotation json file
        output_name: video name
        capture_rate: take screenshot every _ second
        split: train, valid, split ratio
    Returns:
        null
    """
    filename = json_filename.split('.')[0]
    filepath = os.path.join(sa_folder_path, json_filename)
    with open(filepath) as f:
        annotation_data = json.load(f)
    video_url = annotation_data['metadata']['url']
    # Open the video stream
    video_stream = cv2.VideoCapture(video_url)
    length = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    frame_list = [frame_no for frame_no in range(length) if frame_no % (fps * capture_rate) == 0]
    instances = annotation_data['instances']
    for frame_no in frame_list:
        split_folder = split_dataset(split) if split else ''
        image_filename = f"{output_name}/{split_folder}/images/{filename}_{frame_no}.jpg"
        label_filename = f"{output_name}/{split_folder}/labels/{filename}_{frame_no}.txt"
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = video_stream.read()
        cv2.imwrite(image_filename, frame)
        timestamp = (frame_no / fps) * 1000000
        save_yolo_txt(instances, timestamp, frame.shape, label_filename)


def _convert_timestamp(timestamps):
    """
    helper function to merge instance
    Args:
        timestamps:
    Returns:
        merged timestamp instance
    """
    result = []
    for i in range(0, len(timestamps) - 1):
        group = {
            "start": timestamps[i]["timestamp"],
            "end": timestamps[i + 1]["timestamp"],
            "points": timestamps[i]["points"]
        }
        result.append(group)
    return result


def save_yolo_txt(instances, timestamp, shape, filename):
    """
    save txt file for current frame
    Args:
        instances: sa format instant from json file
        timestamp: current frame timestamp
        shape: image shape
        filename: output filename
    Returns:
        null
    """
    # Iterate through instances in the annotation
    with open(filename, 'w') as file:
        for instance in instances:
            for parameter in instance['parameters']:
                if parameter['start'] <= timestamp <= parameter['end']:
                    timestamps_anno = _convert_timestamp(parameter['timestamps'])
                    for anno in timestamps_anno:
                        if anno['start'] <= timestamp <= anno['end']:
                            x1, y1, x2, y2 = (
                                anno['points']['x1'],
                                anno['points']['y1'],
                                anno['points']['x2'],
                                anno['points']['y2']
                            )
                            # Convert annotation to YOLOv5 format
                            frame_height, frame_width, _ = shape
                            yolo_format = f"{CLASSES_DICT[instance['meta']['className']]} {((x1 + x2) / 2) / frame_width} {((y1 + y2) / 2) / frame_height} {(x2 - x1) / frame_width} {(y2 - y1) / frame_height}"
                            # Write line
                            file.write(f"{yolo_format}\n")


if __name__ == "__main__":
    CAPTURE_RATE = 30  # every _ second
    SPLIT = {
        "train": 70,
        "test": 10,
        "val": 20,
    }
    sa_folder_path = r'C:\Users\User\Downloads\msc_solubility_20231116_Nov_16_2023_12_47_Rama'
    generate_yolo(sa_folder_path, capture_rate=CAPTURE_RATE, split=SPLIT)
