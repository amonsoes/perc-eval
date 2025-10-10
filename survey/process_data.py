import os
import torch

from PIL import Image
from process import DatasetConstructor
import torchvision.transforms as T
from torchvision.utils import save_image


def process_data_root():
    dataset_contructor = DatasetConstructor()
    dataset_contructor.clear_directories()
    
    dir_root = './data/perc_eval/'
    ori_image_root = './data/perc_eval/images/'
    target_image_root = './data/perc_eval/images_new/'
    to_tensor = T.ToTensor()
    n = 0
    for pbound in ['4.0', '8.0', '15.0', '23.0', '38.0']:
        for attack in ['pgdl2', 'vmifgsm', 'percal', 'sgd_uap', 'sparsesigmaattack', 'hpf_fgsm', 'square_attack', 'ppba', 'rcw', 'cgnc', 'cgsp', 'dim', 'bsr', 'sga_uap']:
            paths = [( ori_image_root + file, file) for file in os.listdir(dir_root+ pbound + '/' + attack) if file.endswith('.png')]
            images = [(to_tensor(Image.open(path)), file) for path,file in paths]
            ycbcr_images = [to_tensor(Image.open(path).convert('YCbCr')) for path, _ in paths]
            if n % 2 == 0:
                max_dissim_img, _, max_dissim_file = dataset_contructor.frequency_dissimilarity(images)
            else:
                max_dissim_img, _, max_dissim_file = dataset_contructor.clr_int_dissimilarity(ycbcr_images, images)
            n += 1
            save_image(max_dissim_img, f'{target_image_root}{max_dissim_file}')
            print('done')

def gather_pool_of_images():
    imgs_dict = {str(i):[] for i in range(0, 1000)}
    for e,cls in enumerate(classes):
        images, ycbcr_images = self.data.get_images_for_class(cls)
        if e % 2 == 0:
            max_dissim_img, dissim = self.frequency_dissimilarity(images)
        else:
            max_dissim_img, dissim = self.clr_int_dissimilarity(ycbcr_images, images)
        save_image(max_dissim_img, f'{self.survey_dir}{e}.png')
        self.save_cls(filename=e, cls=cls['index'])



if __name__ == '__main__':

    process_data_root()
