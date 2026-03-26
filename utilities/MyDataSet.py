import csv
from abc import ABC
import cv2
import numpy as np
import torch
from torch.utils import data
import nibabel as nib
from scipy import ndimage
from numpy.linalg import inv
import torch.nn.functional as F
from collections import defaultdict
import random
from tqdm import tqdm

__all__ = ['SemiSegDataset']


class SemiSegDataset(data.Dataset, ABC):
    def __init__(self, data_dir, train_data_csv, valid_data_csv, image_size=(224, 224),
                 label_ratio: float = 0.2, mode='train', random_seed: int = 42):
        super(SemiSegDataset, self).__init__()

        self.data_dir = data_dir
        self.train_data_csv = train_data_csv
        self.image_size = image_size
        self.mode = mode
        self.train_data_list = self.file2list(train_data_csv)
        self.valid_data_list = self.file2list(valid_data_csv)
        self.train_data_list_phi=sorted(list(set(self.train_data_list)-set(self.valid_data_list)),key=lambda x:x[0].split('/')[0]) # split train/valid samples
        self.random_seed = random_seed
        self.label_ratio = label_ratio
        random.seed(random_seed)
        if self.mode == 'train':
            self.label_ratio = label_ratio
            num_labeled_subj = int(len(self.train_data_list) * label_ratio)
            if self.label_ratio==1.0:
                self.train_subjects = random.sample(self.train_data_list_phi, len(self.train_data_list_phi))
            else:
                self.train_subjects = random.sample(self.train_data_list_phi, num_labeled_subj)
            if self.label_ratio==1.0:
                self.unlabeled_train_subjects=self.train_subjects
            else:
                self.unlabeled_train_subjects = [subj for subj in self.train_data_list if subj not in self.train_subjects]
            
            
            self.train_label_data = []
            self.train_unlabeled_data = []

            for index in tqdm(range(len(self.train_subjects))):
                whole_img, whole_label = self.getperdata(self.data_dir, self.train_subjects, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)
                    temp_label = label[:, :, i]

                    self.train_label_data.append([temp_image, temp_label])
            for index in tqdm(range(len(self.unlabeled_train_subjects))):
                whole_img, _ = self.getperdata(self.data_dir, self.unlabeled_train_subjects, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)

                    self.train_unlabeled_data.append(temp_image)
        else:
            self.valid_label_data = []
            for index in range(len(self.valid_data_list)):
                whole_img, whole_label = self.getperdata(self.data_dir, self.valid_data_list, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)
                    temp_label = label[:, :, i]

                    self.valid_label_data.append([temp_image, temp_label])

    def __len__(self):
        if self.label_ratio <= 0.5:
            return len(self.train_unlabeled_data) if self.mode == 'train' else len(self.valid_label_data)
        else:
            return len(self.train_label_data) if self.mode == 'train' else len(self.valid_label_data)

    def __getitem__(self, item):
        if self.mode == 'train':
            if self.label_ratio <= 0.5:
                randomd_idx = np.random.randint(low=0, high=len(self.train_label_data))
                # labeled_img = self.train_label_data[item % len(self.train_label_data)][0]
                # labeled_lab = self.train_label_data[item % len(self.train_label_data)][1]
                labeled_img = self.train_label_data[randomd_idx][0]
                labeled_lab = self.train_label_data[randomd_idx][1]
                unlabeled_img = self.train_unlabeled_data[item]

                tensor_labeled_img = torch.Tensor(labeled_img)
                tensor_labeled_lab = torch.LongTensor(labeled_lab)
                tensor_unlabeled_img = torch.Tensor(unlabeled_img)

                return tensor_labeled_img, tensor_labeled_lab, tensor_unlabeled_img
            else:
                randomd_idx = np.random.randint(low=0, high=len(self.train_unlabeled_data))
                labeled_img = self.train_label_data[item][0]
                labeled_lab = self.train_label_data[item][1]
                # unlabeled_img = self.train_unlabeled_data[item % len(self.train_unlabeled_data)]
                unlabeled_img = self.train_unlabeled_data[randomd_idx]

                tensor_labeled_img = torch.Tensor(labeled_img)
                tensor_labeled_lab = torch.LongTensor(labeled_lab)
                tensor_unlabeled_img = torch.Tensor(unlabeled_img)

                return tensor_labeled_img, tensor_labeled_lab, tensor_unlabeled_img
        else:
            labeled_img = self.valid_label_data[item][0]
            labeled_lab = self.valid_label_data[item][1]
            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)

            return tensor_labeled_img, tensor_labeled_lab

    def file2list(self, file_csv):
        subj_dic = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                subj = (line['image_filenames'], line['label_filenames'])
                subj_dic.append(subj)
        return subj_dic

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        image = self.crop_image(image, x_centre, y_centre, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index][0]
        label_name = data_dir + '/' + data_list[index][1]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        nib_label = nib.load(label_name)
        whole_label = nib_label.get_fdata()
        whole_label = whole_label.astype('float32')

        return whole_image, whole_label

    def crop_image(self, image, cx, cy, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        rX = size[0] // 2
        rY = size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_, :]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        label_set=sorted(np.unique(label).tolist())
        for id,lb in enumerate(label_set):
            label[label==lb]=id
        x_centre, y_centre = int(x / 2), int(y / 2)
        label = self.crop_image(label, x_centre, y_centre, image_size)
        return label


class SemiSegDataset_1(data.Dataset, ABC):
    def __init__(self, data_dir, train_data_csv, valid_data_csv, image_size=(224, 224),
                 label_ratio: float = 0.2, mode='train', random_seed: int = 42,shift_label=60,shift_unlabel=30):
        super(SemiSegDataset_1, self).__init__()

        self.data_dir = data_dir
        self.train_data_csv = train_data_csv
        self.image_size = image_size
        self.mode = mode
        self.train_data_list = self.file2list(train_data_csv)
        self.valid_data_list = self.file2list(valid_data_csv)
        self.train_data_list_phi=sorted(list(set(self.train_data_list)-set(self.valid_data_list)),key=lambda x:x[0].split('/')[0]) # split train/valid samples
        self.random_seed = random_seed
        self.label_ratio = label_ratio
        random.seed(random_seed)
        if self.mode == 'train':
            self.label_ratio = label_ratio
            num_labeled_subj = int(len(self.train_data_list) * label_ratio)
            if self.label_ratio==1.0:
                self.train_subjects = random.sample(self.train_data_list_phi, len(self.train_data_list_phi))
            else:
                self.train_subjects = random.sample(self.train_data_list_phi, num_labeled_subj)
            if self.label_ratio==1.0:
                self.unlabeled_train_subjects=self.train_subjects
            else:
                self.unlabeled_train_subjects = [subj for subj in self.train_data_list if subj not in self.train_subjects]

            
            
            self.train_label_data = []
            self.train_unlabeled_data = []

            for index in range(len(self.train_subjects)):
                whole_img, whole_label = self.getperdata(self.data_dir, self.train_subjects, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)
                    temp_label = label[:, :, i]

                    self.train_label_data.append([temp_image, temp_label])
            for index in range(len(self.unlabeled_train_subjects)):
                whole_img, _ = self.getperdata(self.data_dir, self.unlabeled_train_subjects, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)

                    self.train_unlabeled_data.append(temp_image)
        else:
            self.valid_label_data = []
            for index in range(len(self.valid_data_list)):
                whole_img, whole_label = self.getperdata(self.data_dir, self.valid_data_list, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)
                    temp_label = label[:, :, i]

                    self.valid_label_data.append([temp_image, temp_label])

    def __len__(self):
        return len(self.train_label_data) if self.mode == 'train' else len(self.valid_label_data)
    
    def __getitem__(self, item):
        if self.mode == 'train':
            # 从无标签数据中随机选择4个样本的索引
            random_idx = np.random.choice(len(self.train_unlabeled_data), 4, replace=False)
            
            # 获取有标签数据
            labeled_img = self.train_label_data[item][0]
            labeled_lab = self.train_label_data[item][1]
            
            # 根据索引获取无标签数据，结果是一个包含多个 numpy 数组的列表
            unlabeled_img_list = [self.train_unlabeled_data[idx] for idx in random_idx]

            # --- 修改部分 开始 ---
            # 首先将 numpy 数组的列表转换为一个单一的 numpy 数组
            unlabeled_img_array = np.array(unlabeled_img_list)
            # --- 修改部分 结束 ---

            # 将数据转换为 PyTorch 张量
            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)
            
            # --- 修改部分 开始 ---
            # 从转换后的 numpy 数组创建张量
            tensor_unlabeled_img = torch.Tensor(unlabeled_img_array)
            # --- 修改部分 结束 ---

            return tensor_labeled_img, tensor_labeled_lab, tensor_unlabeled_img
        
        else: # for self.mode == 'valid' or 'test'
            labeled_img = self.valid_label_data[item][0]
            labeled_lab = self.valid_label_data[item][1]
            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)

            return tensor_labeled_img, tensor_labeled_lab

    # def __getitem__(self, item):
    #     if self.mode == 'train':
    #         random_idx=np.random.choice(len(self.train_unlabeled_data),4,replace=False)
            
    #         labeled_img=self.train_label_data[item][0]
    #         labeled_lab=self.train_label_data[item][1]
    #         unlabeled_img=[self.train_unlabeled_data[idx] for idx in random_idx]

    #         tensor_labeled_img = torch.Tensor(labeled_img)
    #         tensor_labeled_lab = torch.LongTensor(labeled_lab)
    #         tensor_unlabeled_img = torch.Tensor(unlabeled_img)

    #         return tensor_labeled_img, tensor_labeled_lab, tensor_unlabeled_img
    #     else:
    #         labeled_img = self.valid_label_data[item][0]
    #         labeled_lab = self.valid_label_data[item][1]
    #         tensor_labeled_img = torch.Tensor(labeled_img)
    #         tensor_labeled_lab = torch.LongTensor(labeled_lab)

    #         return tensor_labeled_img, tensor_labeled_lab

    def file2list(self, file_csv):
        subj_dic = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                subj = (line['image_filenames'], line['label_filenames'])
                subj_dic.append(subj)
        return subj_dic

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        image = self.crop_image(image, x_centre, y_centre, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index][0]
        label_name = data_dir + '/' + data_list[index][1]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        nib_label = nib.load(label_name)
        whole_label = nib_label.get_fdata()
        whole_label = whole_label.astype('float32')

        return whole_image, whole_label

    def crop_image(self, image, cx, cy, size, constant_values=0,shift_value=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        rX = size[0] // 2
        rY = size[1] // 2
        
        cx=cx+shift_value
        cy=cy+shift_value
        
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_, :]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop
    
    def label_preprocessing(self, label, image_size):
        # 3D数据包含所有的类
        x, y, z = label.shape
        label_set=sorted(np.unique(label).tolist())
        for id,lb in enumerate(label_set):
            label[label==lb]=id
        x_centre, y_centre = int(x / 2), int(y / 2)
        label = self.crop_image(label, x_centre, y_centre, image_size)
        return label

    # def label_preprocessing(self, label, image_size):
    #     x, y, z = label.shape
    #     x_centre, y_centre = int(x / 2), int(y / 2)
    #     label = self.crop_image(label, x_centre, y_centre, image_size)
    #     return label


class SemiSegDataset_2(data.Dataset, ABC):
    def __init__(self, data_dir, train_data_csv, valid_data_csv, image_size=(224, 224),
                 label_ratio: float = 0.2, mode='train', random_seed: int = 42,shift_label=0,shift_unlabel=0):
        super(SemiSegDataset_2, self).__init__()

        self.data_dir = data_dir
        self.train_data_csv = train_data_csv
        self.image_size = image_size
        self.mode = mode
        self.train_data_list = self.file2list(train_data_csv)
        self.valid_data_list = self.file2list(valid_data_csv)
        self.train_data_list_phi=sorted(list(set(self.train_data_list)-set(self.valid_data_list)),key=lambda x:x[0].split('/')[0]) # split train/valid samples
        self.random_seed = random_seed
        self.label_ratio = label_ratio
        self.shift_label=shift_label
        self.shift_unlabel=shift_unlabel
        random.seed(random_seed)
        if self.mode == 'train':
            self.label_ratio = label_ratio
            num_labeled_subj = round(len(self.train_data_list) * label_ratio)
            if self.label_ratio==1.0:
                self.train_subjects = random.sample(self.train_data_list_phi, len(self.train_data_list_phi))
            else:
                self.train_subjects = random.sample(self.train_data_list_phi, num_labeled_subj)
            if self.label_ratio==1.0:
                self.unlabeled_train_subjects=self.train_subjects
            else:
                self.unlabeled_train_subjects = [subj for subj in self.train_data_list if subj not in self.train_subjects]

            
            
            self.train_label_data = []
            self.train_unlabeled_data = []

            for index in range(len(self.train_subjects)):
                whole_img, whole_label = self.getperdata(self.data_dir, self.train_subjects, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)
                    temp_label = label[:, :, i]

                    self.train_label_data.append([temp_image, temp_label])
            for index in range(len(self.unlabeled_train_subjects)):
                whole_img, _ = self.getperdata(self.data_dir, self.unlabeled_train_subjects, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)

                    self.train_unlabeled_data.append(temp_image)
        else:
            self.valid_label_data = []
            for index in range(len(self.valid_data_list)):
                whole_img, whole_label = self.getperdata(self.data_dir, self.valid_data_list, index)
                image = self.data_preprocessing(whole_img, image_size=self.image_size)
                label = self.label_preprocessing(whole_label, image_size=self.image_size)

                for i in range(image.shape[-1]):
                    temp_image = np.expand_dims(image[:, :, i], axis=0)
                    temp_label = label[:, :, i]

                    self.valid_label_data.append([temp_image, temp_label])

    def __len__(self):
        return len(self.train_label_data) if self.mode == 'train' else len(self.valid_label_data)
    
    def __getitem__(self, item):
        if self.mode == 'train':
            # 从无标签数据中随机选择4个样本的索引
            random_idx = np.random.choice(len(self.train_unlabeled_data), 4, replace=False)
            
            shift_label=np.random.randint(-self.shift_label,self.shift_label)
            # 获取有标签数据
            labeled_img = self.crop_image(self.train_label_data[item][0],size=self.image_size,shift_value=shift_label)
            labeled_lab = self.crop_image(self.train_label_data[item][1],size=self.image_size,shift_value=shift_label)

            
            shift_unlabel=np.random.randint(-self.shift_unlabel,self.shift_unlabel,4).tolist()
            # 根据索引获取无标签数据，结果是一个包含多个 numpy 数组的列表
            unlabeled_img_list = [self.crop_image(self.train_unlabeled_data[idx],size=self.image_size,shift_value=shift) for idx,shift in zip(random_idx,shift_unlabel)]

            # --- 修改部分 开始 ---
            # 首先将 numpy 数组的列表转换为一个单一的 numpy 数组
            unlabeled_img_array = np.array(unlabeled_img_list)
            # --- 修改部分 结束 ---

            # 将数据转换为 PyTorch 张量
            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)
            
            # --- 修改部分 开始 ---
            # 从转换后的 numpy 数组创建张量
            tensor_unlabeled_img = torch.Tensor(unlabeled_img_array)
            # --- 修改部分 结束 ---

            return tensor_labeled_img, tensor_labeled_lab, tensor_unlabeled_img
        
        else: # for self.mode == 'valid' or 'test'
            labeled_img = self.valid_label_data[item][0]
            labeled_lab = self.valid_label_data[item][1]
            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)

            return tensor_labeled_img, tensor_labeled_lab


    def file2list(self, file_csv):
        subj_dic = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                subj = (line['image_filenames'], line['label_filenames'])
                subj_dic.append(subj)
        return subj_dic

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index][0]
        label_name = data_dir + '/' + data_list[index][1]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata()

        whole_image = whole_image.astype('float32')

        nib_label = nib.load(label_name)
        whole_label = nib_label.get_fdata()
        whole_label = whole_label.astype('float32')

        return whole_image, whole_label

    def crop_image(self, image, size, constant_values=0,shift_value=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        if image.ndim==3:
            image = np.transpose(image, [1, 2, 0])
        X, Y = image.shape[:2]
        rX = size[0] // 2
        rY = size[1] // 2

        cx = X // 2 + shift_value
        cy = Y // 2 + shift_value

        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 2:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        
        if image.ndim==3:
            crop = np.transpose(crop, [2, 0, 1])
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        label_set=sorted(np.unique(label).tolist())
        for id,lb in enumerate(label_set):
            label[label==lb]=id
        x_centre, y_centre = int(x / 2), int(y / 2)
        return label


class FullyDataset(SemiSegDataset):
    def __init__(self, *args, **kwargs):
        super(FullyDataset, self).__init__(*args, **kwargs)

    def __len__(self):
        return len(self.train_label_data) if self.mode == 'train' else len(self.valid_label_data)

    def __getitem__(self, item):
        if self.mode == 'train':
            labeled_img = self.train_label_data[item][0]
            labeled_lab = self.train_label_data[item][1]

            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)

            return tensor_labeled_img, tensor_labeled_lab
        else:
            labeled_img = self.valid_label_data[item][0]
            labeled_lab = self.valid_label_data[item][1]
            tensor_labeled_img = torch.Tensor(labeled_img)
            tensor_labeled_lab = torch.LongTensor(labeled_lab)

            return tensor_labeled_img, tensor_labeled_lab
