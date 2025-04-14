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
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        self.labels = pd.read_csv(path_labels)
        self.img_dir = path_images
        self.transform = T.Compose([T.Resize(256,  interpolation=InterpolationMode.BILINEAR),
                                    T.CenterCrop(224),
                                    T.ToTensor()])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        image = self.transform(image)
        label = self.labels.iloc[idx, 6] - 1
        return image, label, img_path.split('/')[-1]





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
    
