import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img

def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
        post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
        label_path = os.path.join(self.dataset_path, 'GT', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if 'train' in self.data_pro_type:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index] + '.png')
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index] + '.png')
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index] + '.png')
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index] + '.png')
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index] + '.png')
        else:
            pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
            post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
            T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
            T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
            cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)

class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label, clf_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label
    
    def convert_color_to_class(self, clf_label):
        """
        Converts a color-coded label image to class indices dynamically.
        This assumes that unique colors in the image are predefined.
        """
        color_to_class = {
            (0, 0, 0): 0,         # Background
            (255, 0, 0): 1,       # No damage (Red)
            (0, 255, 0): 2,       # Minor damage (Green)
            (0, 0, 255): 3,       # Major damage (Blue)
            (255, 255, 0): 4      # Destroyed (Yellow)
        }

        # Convert to indices
        h, w, c = clf_label.shape
        clf_label_indices = np.zeros((h, w), dtype=np.uint8)

        # Find unique colors and map them to indices
        for color, class_idx in color_to_class.items():
            mask = (clf_label[:, :, 0] == color[0]) & (clf_label[:, :, 1] == color[1]) & (clf_label[:, :, 2] == color[2])
            clf_label_indices[mask] = class_idx

        return clf_label_indices

    def __getitem__(self, index):
        # For 'train' dataset
        if self.data_pro_type == 'train':
            parts = self.data_list[index]

            pre_img_name = f"{parts}_pre_disaster.png"
            post_img_name = f"{parts}_post_disaster.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'target', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'target', post_img_name)

            pre_img = self.loader(pre_path)
            post_img = self.loader(post_path)
            loc_label = self.loader(loc_label_path)
            
            # Check if the label is 3D (height, width, channels) or 2D (height, width)
            if len(loc_label.shape) == 3:  # 3D data (height, width, channels)
                loc_label = loc_label[:, :, 0]  # Take the first channel
                loc_label = (loc_label == 255).astype(int)
            # If it's already 2D (single channel), use it directly
            elif len(loc_label.shape) == 2:  # 2D data (height, width)
                pass  # No need to do anything, it's already the expected format
            
            # Load classifier label and convert colors to class indices
            clf_label = self.loader(clf_label_path)

            if len(clf_label.shape) == 3:
                clf_label = self.convert_color_to_class(clf_label)

            

            # Apply augmentation transforms during training
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)

            # Modify class labels during training (optional adjustment)
            clf_label[clf_label == 0] = 255  # Adjust the background label (optional)

        # For 'val' dataset
        elif self.data_pro_type == 'val':
            parts = self.data_list[index]

            pre_img_name = f"{parts}_pre_disaster.png"
            post_img_name = f"{parts}_post_disaster.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            loc_label_path = os.path.join(self.dataset_path, 'target', pre_img_name)
            clf_label_path = os.path.join(self.dataset_path, 'target', post_img_name)

            pre_img = self.loader(pre_path)
            post_img = self.loader(post_path)
            loc_label = self.loader(loc_label_path)

            # Check if the label is 3D (height, width, channels) or 2D (height, width)
            if len(loc_label.shape) == 3:  # 3D data (height, width, channels)
                loc_label = loc_label[:, :, 0]  # Take the first channel
                loc_label = (loc_label == 255).astype(int)
            # If it's already 2D (single channel), use it directly
            elif len(loc_label.shape) == 2:  # 2D data (height, width)
                pass  # No need to do anything, it's already the expected format
            
            # Load classifier label and convert colors to class indices
            clf_label = self.loader(clf_label_path)

            if len(clf_label.shape) == 3:
                clf_label = self.convert_color_to_class(clf_label)

            # No augmentation during validation
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)

            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)

        # For 'test' dataset type (no label available)
        else:  # 'test' dataset type
            parts = self.data_list[index]
            
            pre_img_name = f"{parts}_pre_disaster.png"
            post_img_name = f"{parts}_post_disaster.png"

            pre_path = os.path.join(self.dataset_path, 'images', pre_img_name)
            post_path = os.path.join(self.dataset_path, 'images', post_img_name)
            
            # No labels for the test set, just images
            pre_img = self.loader(pre_path)
            post_img = self.loader(post_path)

            pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
            pre_img = np.transpose(pre_img, (2, 0, 1))

            post_img = imutils.normalize_img(post_img)  # imagenet normalization
            post_img = np.transpose(post_img, (2, 0, 1))

            # Return image data along with the name of the image (index or file name)
            return pre_img, post_img, parts

        # Return data for training or validation, which includes images and labels
        return pre_img, post_img, loc_label, clf_label, parts

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU' in args.dataset:
        dataset = ChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    elif 'xBD' in args.dataset:
        dataset = DamageAssessmentDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=6,
                                 drop_last=False)
        return data_loader
    
    elif 'SECOND' in args.dataset:
        dataset = SemanticChangeDetectionDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
