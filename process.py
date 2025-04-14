import json
import random
import torchvision.transforms as T
import pandas as pd
import os

from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


class Nips17Subset(Dataset):
    
    def __init__(self):
        super().__init__()
        self.dir = './data/ext_nips17/'
        self.wnid_to_class = self.get_wn2id()
        self.file_to_wnid = self.get_image_ids()
        self.ind_to_file = {str(e):file for e, file in enumerate(self.file_to_wnid.keys())}
        self.transform = T.Compose([T.Resize(256,  interpolation=InterpolationMode.BILINEAR),
                                    T.CenterCrop(224),
                                    T.ToTensor()])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        filename = self.ind_to_file[str(idx)]
        wn_id = self.file_to_wnid[filename]
        image = Image.open(self.dir + '/' + wn_id + '/' +   filename)
        image = self.transform(image)
        label = self.wnid_to_class[wn_id]
        return image, label, filename

    def get_wn2id(self):
        with open('./data/ext_nips17/imagenet_class_index.json', 'r') as f:
            classes_json = json.loads(f.read())
        wnid_classes = {wn_id[0] : cls for cls, wn_id  in classes_json.items()}
        return wnid_classes

    def get_image_ids(self):
        file_2_wnid = {}
        for root, dirs, files in os.walk(self.dir, topdown=False):
            for file in files:
                if file.endswith('.JPEG'):
                    file_2_wnid[file] = root.split('/')[-1]
        return file_2_wnid

        





########## semantically diverse functions




def get_node_level_and_children(num_classes):

    with open('./hierarchy.json', 'r') as f:
        classes_json = json.loads(f.read())
        nodes = recursive_depth([], [classes_json], num_classes)
        return nodes


def recursive_depth(leafs, classes_json, num_classes, leaf_incl_prob=0.99):

    call_stack = []
    for node in classes_json:
        children = node['children']

        for child in children:
            if not child.get('children', 0):
                choice = random.choices([0,1], weights=[10*leaf_incl_prob, 10*(1-leaf_incl_prob)])
                if choice == [1]:
                    leafs.append(child)
            else:
                call_stack.append(child)
    
    if len(call_stack) + len(leafs) >= num_classes:
        leafs.extend(call_stack)
        return leafs
    else:
        leaf_incl_prob -= 0.05
        leafs = recursive_depth(leafs, classes_json=call_stack, num_classes=num_classes, leaf_incl_prob=leaf_incl_prob)
        return leafs
    
def get_classes_from_nodes(nodes):
    leafs = []
    for node in nodes:
        rand_leaf = rand_walk_node_recursively(node)
        leafs.append(rand_leaf)
    return leafs


def rand_walk_node_recursively(node):
    if not node.get('children', 0):
        return node
    else:
        rand_sub_node = random.choice(node['children'])
        sub_node = rand_walk_node_recursively(rand_sub_node)
    return sub_node


####################

def gather_pool_of_images(classes):
    imgs_dict = {str(i):[] for i in range(0, 1000)}
    data = Nips17Subset()

    for img, label, path in data:
        imgs_dict[str(label)].append(img)
    
    print('done')
    



if __name__ == '__main__':

    ## get semantic classes
    nodes = get_node_level_and_children(50)
    classes = get_classes_from_nodes(nodes)

    ## get images
    gather_pool_of_images(classes)
    
