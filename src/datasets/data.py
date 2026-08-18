import csv

from torch.utils.data import DataLoader, Subset, random_split
from src.datasets.subsets import Nips17Subset, CustomCIFAR10, CustomCIFAR100, FlickrSubset, SurveySubset
from src.datasets.data_transforms.img_transform import PreTransforms, PostTransforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

class Data:
    
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset = self.loader(dataset_name, *args, **kwargs)
    
    def loader(self, dataset_name, *args, **kwargs):
        if dataset_name == 'nips17':
            dataset = Nips17ImgNetData(*args, **kwargs)
        elif dataset_name == 'mnist':
            dataset = MNISTDataset(*args, **kwargs)
        elif dataset_name == 'cifar10':
            dataset = CIFAR10Dataset(*args, **kwargs)
        elif dataset_name == '140k_flickr_faces':
            dataset = FlickrFaces(*args, **kwargs)
        elif dataset_name == 'survey':
            dataset = SurveyDataset(*args, **kwargs)
        else:
            raise ValueError('Dataset not recognized')
        return dataset

class BaseDataset:
    
    def __init__(self,
                dataset_name,
                model,
                device,
                batch_size,
                transform,
                adversarial_opt,
                adversarial_training_opt,
                jpeg_compression,
                jpeg_compression_rate,
                target_transform=None,
                input_size=224):


        self.transform_type = transform
        self.transforms = PreTransforms(device=device,
                                    target_transform=target_transform, 
                                    input_size=input_size, 
                                    dataset_type=dataset_name,
                                    adversarial_opt=adversarial_opt)
        self.post_transforms = PostTransforms(transform,
                                            adversarial_opt=adversarial_opt,
                                            input_size=input_size,
                                            jpeg_compression=jpeg_compression,
                                            jpeg_compression_rate=jpeg_compression_rate,
                                            dataset_type=dataset_name,
                                            model=model,
                                            device=device,
                                            val_base_trm_size=self.transforms.val_base_trm_size)
        self.adversarial_opt = adversarial_opt
        self.adversarial_training_opt = adversarial_training_opt
        self.device = device
        self.batch_size = batch_size
        self.x, self.y = input_size, input_size

class FlickrFaces(BaseDataset):

    def __init__(self, n_datapoints, *args,**kwargs):
        super().__init__('140k_flickr_faces', *args, **kwargs)
        
        self.dataset_type = '140k_flickr_faces'

        train_data, val_data, test_data = self.get_data()
        if n_datapoints != -1:
            train_data, _ = random_split(train_data, [n_datapoints, len(train_data)-n_datapoints])
            val_data, _ = random_split(val_data, [n_datapoints, len(val_data)-n_datapoints])
            dataset_size = len(test_data)
            split_size = int(dataset_size * 0.5)
            indices = list(range(dataset_size))
            subset_indices = indices[split_size:]
            test_data = Subset(test_data, subset_indices)
            test_data, _ = random_split(test_data, [n_datapoints, len(test_data)-n_datapoints])
        else:
            dataset_size = len(test_data)
            split_size = int(dataset_size * 0.5)
            indices = list(range(dataset_size))
            subset_indices = indices[split_size:]
            test_data = Subset(test_data, subset_indices)
        self.train = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.validation = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        self.test = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)


    def get_data(self):
        path = './data/140k_flickr_faces'

        train_labels = path + '/train.csv'
        val_labels = path + '/valid.csv'
        test_labels = path + '/test.csv'
        data_path = path + '/real_vs_fake/real-vs-fake/'


        train = FlickrSubset(label_path=train_labels, 
                img_path=data_path, 
                transform=self.transforms.transform_train, 
                target_transform=self.transforms.target_transform, 
                adversarial=self.adversarial_opt.adversarial)
        val = FlickrSubset(label_path=val_labels,
                img_path=data_path, 
                transform=self.transforms.transform_val, 
                target_transform=self.transforms.target_transform, 
                adversarial=self.adversarial_opt.adversarial)
        test = FlickrSubset(label_path=test_labels, 
                img_path=data_path, 
                transform=self.transforms.transform_val, 
                target_transform=self.transforms.target_transform, 
                adversarial=self.adversarial_opt.adversarial,
                is_test_data=True)


        return train, val, test


class Nips17ImgNetData(BaseDataset):

    def __init__(self, n_datapoints, *args,**kwargs):
        super().__init__('nips17', *args, **kwargs)
        
        self.categories = self.get_categories()
        self.dataset_type = 'nips17'

        self.test_data = self.get_data(transform_val=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform)
        if n_datapoints == -1:
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)

    def get_data(self, transform_val, target_transform):
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        test = Nips17Subset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test
        
    def get_categories(self):
        categories = {}
        path = './data/nips17/categories.csv'
        with open(path, 'r') as cats:
            filereader = csv.reader(cats)
            next(filereader)
            for ind, cat in filereader:
                categories[int(ind) - 1] = cat
        return categories

class SurveyDataset(Nips17ImgNetData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist_dict = {


            'percal':{'4.0':[12],
                      '8.0':[18],
                      '15.0':[10],
                      '23.0':[3],
                      '38.0':[5]},
            'square':{'4.0':[20],
                      '8.0':[27],
                      '15.0':[28],
                      '23.0':[32],
                      '38.0':[36]},
            'ppba':{'4.0':[40],
                      '8.0':[44],
                      '15.0':[49],
                      '23.0':[53],
                      '38.0':[56]},
            'rcw':{'4.0':[77],
                      '8.0':[66],
                      '15.0':[71],
                      '23.0':[74],
                      '38.0':[63]},
            'cgnc':{'4.0':[83],
                      '8.0':[85],
                      '15.0':[88],
                      '23.0':[93],
                      '38.0':[97]},
            'cgsp':{'4.0':[102],
                      '8.0':[107],
                      '15.0':[109],
                      '23.0':[112],
                      '38.0':[117]},
            'sparsesigmaattack':{'4.0':[120],
                      '8.0':[127],
                      '15.0':[131],
                      '23.0':[134],
                      '38.0':[139]},
            'hpf_fgsm':{'4.0':[142],
                      '8.0':[145],
                      '15.0':[150],
                      '23.0':[153],
                      '38.0':[157]},
            'dim':{'4.0':[161],
                      '8.0':[165],
                      '15.0':[168],
                      '23.0':[175],
                      '38.0':[179]},
            'bsr':{'4.0':[180],
                      '8.0':[184],
                      '15.0':[188],
                      '23.0':[192],
                      '38.0':[196]},
            'vmifgsm':{'4.0':[202],
                      '8.0':[205],
                      '15.0':[209],
                      '23.0':[214],
                      '38.0':[219]},
            'pgdl2':{'4.0':[220],
                      '8.0':[224],
                      '15.0':[228],
                      '23.0':[235],
                      '38.0':[238]},
            'sgd_uap':{'4.0':[241],
                      '8.0':[245],
                      '15.0':[248],
                      '23.0':[253],
                      '38.0':[258]},
            'sga_uap':{'4.0':[263],
                      '8.0':[265],
                      '15.0':[271],
                      '23.0':[275],
                      '38.0':[279]},

        }

    def get_data(self, transform_val, target_transform):
        path_test = './data/survey_images/'
        path_labels = path_test + 'images/images.csv'
        path_images = path_test + 'images/'
        test = SurveySubset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test



class MNISTDataset(BaseDataset):
    
    def __init__(self,
                n_datapoints,
                *args,
                **kwargs):
        super().__init__('mnist', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'mnist'
        
        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        if n_datapoints == -1:
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    
    def get_data(self):
        train_val_data = MNIST(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = MNIST(root='./data', train=False, download=True, transform=self.transforms.transform_val)
        return train_val_data, test_data
    

class CIFAR10Dataset(BaseDataset):
    
    def __init__(self,
                 n_datapoints,
                *args,
                **kwargs):
        super().__init__('cifar10', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar10'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            train_size = int(len(self.train_val_data) * 0.8) # 80% training data
            valid_size = len(self.train_val_data) - train_size # 20% validation data
            self.train_data, self.val_data = random_split(self.train_val_data, [train_size, valid_size])
            
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        if n_datapoints == -1:
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR10(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = CustomCIFAR10(root='./data', train=False, download=True, transform=self.transforms.transform_val, adversarial=self.adversarial_opt.adversarial, is_test_data=True)
        return train_val_data, test_data


class CIFAR100Dataset(BaseDataset):
    
    def __init__(self,
                *args,
                **kwargs):
        super().__init__('cifar100', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar100'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = random_split(self.train_val_data, [0.8, 0.2])
            # self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR100(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = CustomCIFAR100(adversarial=self.adversarial_opt.adversarial, is_test_data=True, root='./data', train=False, download=True, transform=self.transforms.transform_val)
        return train_val_data, test_data
        
        
if __name__ == '__main__':
    pass
