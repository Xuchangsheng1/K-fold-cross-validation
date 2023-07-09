import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

classname_to_id = {'cat': 1, 'dog': 2}

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # json2COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            # print(json_path)
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1

        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # Build categories
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # The Image field of Coco
    def _image(self, obj, path):
        image = {}
        # from labelme import utils
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # h, w = img_x.shape[:-1]
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # Build the Annotation field of Coco
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # Read the json file and return a JSON object
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # Coco format: [x1, y1, w, h] Corresponding to the Bbox format of coco
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':

    labelme_path = "/media/disk/dataset_singleDir"
    # Random split validation set
    # Obtain a list of all job files in the images directory
    # json_list_path = glob.glob(labelme_path + "/*/*/*.json")
    json_list_path = glob.glob(labelme_path + "/*.json")
    # print(json_list_path)
    skf = KFold(n_splits=5, shuffle=True, random_state=10)
    # X is the feature set and y is the target
    index = 0
    for train_index, test_index in skf.split(json_list_path):
        print("Train:", len(train_index), "Validation:", len(test_index))
        # train_path = []
        # val_path = []
        # for i in train_index:
        #     train_path.append(json_list_path[i])
        # for i in test_index:
        #     val_path.append(json_list_path[i])
        train_path, val_path = np.array(json_list_path)[train_index], np.array(json_list_path)[test_index]
        # print(train_path)
        # # Data division, there is no distinction between val2017 and train2017 directory, all pictures are placed in the Images directory
        # print("train_n:", len(train_path), 'val_n:', len(val_path))
        index += 1
        saved_coco_path = "/media/disk/dataset_singleDir_KFold_r10/5-Fold-"+str(index)+'_'
        # print(saved_coco_path)
        # Create file
        if not os.path.exists("%scoco/annotations/" % saved_coco_path):
            os.makedirs("%scoco/annotations/" % saved_coco_path)
        if not os.path.exists("%scoco/images/train/" % saved_coco_path):
            os.makedirs("%scoco/images/train" % saved_coco_path)
        if not os.path.exists("%scoco/images/val/" % saved_coco_path):
            os.makedirs("%scoco/images/val" % saved_coco_path)
        # if not os.path.exists("%scoco/images/test/" % saved_coco_path):
        #     os.makedirs("%scoco/images/test" % saved_coco_path)
        # Convert the training set to the JSON format of COCO
        l2c_train = Lableme2CoCo()
        # print(l2c_train)
        train_instance = l2c_train.to_coco(train_path)
        l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
        for file in train_path:
            shutil.copy(file.replace("json", "jpg"), "%scoco/images/train/" % saved_coco_path)

        # Transform the val set to the JSON format of Coco
        l2c_val = Lableme2CoCo()
        val_instance = l2c_val.to_coco(val_path)
        l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)
        for file in val_path:
            shutil.copy(file.replace("json", "jpg"), "%scoco/images/val/" % saved_coco_path)
