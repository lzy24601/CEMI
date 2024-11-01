# --- utf-8 ---
# --- function: 将Labeling标注的格式转化为Labelme标注格式 ---

import os
import glob
import shutil
import xml.etree.ElementTree as ET
import json


def get(root, name):
    return root.findall(name)


# 检查读取xml文件是否出错
def get_and_check(root, name, length):
    print(name)
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not fing %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_file, json_file, save_dir, name):
    # 定义通过Labelme标注后生成的json文件
    json_dict = {"version": "4.6.0", "flags": {}, "shapes": [], "imagePath": "", "imageData": None,
                 "imageHeight": 0, "imageWidth": 0}

    # img_name = xml_file.split('.')[0]
    img_path = name + '.jpg'

    json_dict["imagePath"] = img_path

    tree = ET.parse(xml_file)  # 读取xml文件

    root = tree.getroot()

    size = get_and_check(root, 'imagesize', 1)  # 读取xml中<>size<>字段中的内容

    # 获取图片的长宽信息
    width = int(get_and_check(size, 'nrows', 1).text)
    height = int(get_and_check(size, 'ncols', 1).text)

    json_dict["imageHeight"] = height
    json_dict["imageWidth"] = width

    # 当标注中有多个目标时全部读取出来
    for obj in get(root, 'object'):
        # 定义图片的标注信息
        img_mark_inf = {"label": "", "points": [], "group_id": None, "shape_type": "rectangle", "flags": {}}

        category = get_and_check(obj, 'name', 1).text  # 读取当前目标的类别

        img_mark_inf["label"] = category

        bndbox = get_and_check(obj, 'bndbox', 1)  # 获取标注宽信息

        xmin = float(get_and_check(bndbox, 'xmin', 1).text)
        ymin = float(get_and_check(bndbox, 'ymin', 1).text)
        xmax = float(get_and_check(bndbox, 'xmax', 1).text)
        ymax = float(get_and_check(bndbox, 'ymax', 1).text)

        img_mark_inf["points"].append([xmin, ymin])
        img_mark_inf["points"].append([xmax, ymax])
        # print(img_mark_inf["points"])

        json_dict["shapes"].append(img_mark_inf)

    # print("{}".format(json_dict))
    save = save_dir + json_file  # json文件的路径地址

    json_fp = open(save, 'w')  #
    json_str = json.dumps(json_dict)

    json_fp.write(json_str)  # 保存
    json_fp.close()

    # print("{}, {}".format(width, height))


def do_transformation(xml_dir, save_path):
    for fname in os.listdir(xml_dir):
        name = fname.split(".")[0]  # 获取图片名字

        path = os.path.join(xml_dir, fname)  # 文件路径

        save_json_name = name + '.json'
        print(path, save_json_name, save_path, name)
        convert(path, save_json_name, save_path, name)


if __name__ == '__main__':
    img_path = r"./dataset1/save images"
    xml_path = r"./dataset1/dataset 1-10"

    save_json_path = r"./dataset1/json"

    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)

    do_transformation(xml_path, save_json_path)
    # xml = "2007_000039.xml"
    # xjson = "2007_000039.json"

    # convert(xml, xjson)
