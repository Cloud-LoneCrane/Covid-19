import os
import cv2
import numpy as np
import SimpleITK as sitk

class Covid19:
    def __init__(self, dir, batch_size=32, split=0.85, random=True, augment=True, repeat=True, noisy=False):
        """
        :param dir: 存放三个文件夹的文件目录，
        例如是dir="D:/data/COVID19/", COVID19文件夹下面有三个文件夹：rp_im,rp_lung_msk,rp_msk
        :param batch_size: 批处理大小
        :param split: 训练集和测试集的比例
        :param random: 是否随机化
        :param augment: 是否做数据增强
        :param repeat: 是否重复
        :param noisy: 是否添加噪声
        """
        self.__batch_size = batch_size
        self.__idx = 0  # 记录当前返回数据的下标

        self.__normal_imgs = None
        self.__masks = None
        self.__lungs = None

        self.__img_dir = None
        self.__mask_dir = None
        self.__lung_mask_dir = None

        self.__img_files = None
        self.__mask_files = None
        self.__lung_files = None
        self.__create_full_filenames(dir)

        self.__images = None
        self.__masks = None
        self.__lung_masks = None

        self.__test_imgs = None
        self.__test_masks = None
        self.__test_lung_masks = None

        self.length = 0
        self.__construct(dir, split)
        self.__augment(augment, random)
        self.__random(random)
        # print("num imgs:", len(self.__images),
        #       "num masks:", len(self.__masks),
        #       "num lung_mask:", len(self.__lung_masks)
        #       )

    def next_batch(self, batch_size=None):
        rt_imgs = None
        rt_masks = None
        rt_lung_masks = None
        if batch_size is None:
            batch_size = self.__batch_size

        if self.__idx > self.length-batch_size:
            print("COVID19 idx:", self.__idx, " length:", self.length)
            rt_imgs = self.__images[-batch_size:]
            rt_masks = self.__masks[-batch_size:]
            rt_lung_masks = self.__lung_masks[-batch_size]
            self.__idx = 0
            self.__random()
        else:
            rt_imgs = self.__images[self.__idx:self.__idx+batch_size]
            rt_masks = self.__masks[self.__idx:self.__idx+batch_size]
            rt_lung_masks = self.__lung_masks[self.__idx:self.__idx+batch_size]
            self.__idx = self.__idx + batch_size

        return rt_imgs, rt_masks, rt_lung_masks

    @property
    def train_data(self):
        return self.__images, self.__masks, self.__lung_masks

    @property
    def train_data_img_mask(self):
        return self.__images, self.__masks

    @property
    def train_data_img_lungmask(self):
        return self.__images, self.__lung_masks

    @property
    def test_data(self):
        return self.__test_imgs, self.__test_masks, self.__test_lung_masks

    @property
    def test_data_img_mask(self):
        return self.__test_imgs, self.__test_masks

    @property
    def test_data_img_lungmask(self):
        return self.__test_imgs, self.__test_lung_masks

    def __augment(self, augment, random):
        def aug(images):
            aug_imgs = []
            for img in images:
                aug_imgs.append(img)
                aug_imgs.append(img[-1::-1, :])
                aug_imgs.append(img[-1::-1, :][:, -1::-1])
                aug_imgs.append(img[:, -1::-1])
                aug_imgs.append(img.transpose())
                aug_imgs.append(img.transpose()[-1::-1, :])
                aug_imgs.append(img.transpose()[:, -1::-1])
                aug_imgs.append(img.transpose()[:, -1::-1][-1::-1, :])
            return np.array(aug_imgs)

        if augment:
            self.__images = aug(self.__images)
            self.__masks = aug(self.__masks)
            self.__lung_masks = aug(self.__lung_masks)

            self.length = len(self.__images)
            self.__random(random)

    def __random(self, random=True):
        if random:
            permutation = np.random.permutation(self.length)
            self.__images = self.__images[permutation]
            self.__masks = self.__masks[permutation]
            self.__lung_masks = self.__lung_masks[permutation]

    def __construct(self, dir, split):
        def Normalization(imgs):
            """imgs:[batch, h, w]"""
            # 数据归一化到[-1, 1]
            Min, Max = np.min(imgs), np.max(imgs)
            imgs = ((imgs - Min) / (Min - Max) - 0.5) * 2
            return imgs

        total_imgs = []
        total_masks = []
        total_lung_masks = []
        for imgs_file, mask_file, lung_mask_file in zip(self.__img_files, self.__mask_files, self.__lung_mask_files):
            imgs_file_full_path = os.path.join(self.__img_dir, imgs_file)
            mask_file_full_path = os.path.join(self.__mask_dir, mask_file)
            lung_mask_file_full_path = os.path.join(self.__lung_mask_dir, lung_mask_file)
            imgs = sitk.ReadImage(imgs_file_full_path)
            masks = sitk.ReadImage(mask_file_full_path)
            lung_masks = sitk.ReadImage(lung_mask_file_full_path)
            # 1. image转成array，且从[depth, width, height]转成[width, height, depth]
            # 2. deep = batach
            imgs = sitk.GetArrayFromImage(imgs)
            masks = sitk.GetArrayFromImage(masks)
            lung_masks = sitk.GetArrayFromImage(lung_masks)

            total_imgs.append(imgs)
            total_masks.append(masks)
            total_lung_masks.append(lung_masks)

        total_imgs = np.array(total_imgs)
        total_masks = np.array(total_masks)
        total_lung_masks = np.array(total_lung_masks)
        t_imgs = np.concatenate([imgs_element for imgs_element in total_imgs], axis=0)
        self.__masks = np.concatenate([masks_element for masks_element in total_masks], axis=0)
        self.__lung_masks = np.concatenate([lung_masks_element for lung_masks_element in total_lung_masks], axis=0)
        self.__images = Normalization(t_imgs)

        # 3. 将630*630 resize成512*512
        # 每张图片单独resize
        self.__images = [cv2.resize(img, (512, 512)) for img in self.__images]
        self.__masks = [cv2.resize(mask, (512, 512)) for mask in self.__masks]
        self.__lung_masks = [cv2.resize(lung_mask, (512, 512)) for lung_mask in self.__lung_masks]

        idx = int(len(t_imgs)*split)
        self.__test_imgs = self.__images[idx:]
        self.__test_masks = self.__masks[idx:]
        self.__test_lung_masks = self.__lung_masks[idx:]

        self.__images = self.__images[:idx]
        self.__masks = self.__masks[:idx]
        self.__lung_masks = self.__lung_masks[:idx]

        self.length = len(t_imgs)

    def __create_full_filenames(self, dir):
        self.__img_dir = os.path.join(dir, "rp_im")
        self.__mask_dir = os.path.join(dir, 'rp_msk')
        self.__lung_mask_dir = os.path.join(dir, 'rp_lung_msk')
        self.__img_files = os.listdir(self.__img_dir)
        self.__mask_files = os.listdir(self.__mask_dir)
        self.__lung_mask_files = os.listdir(self.__lung_mask_dir)


if __name__ == "__main__":
    # COVID19类的使用示例
    batch_size = 32
    covid19 = Covid19('D:\\data\\COVID19\\', batch_size=batch_size)
    x, mask, lung_mask = covid19.next_batch(2)
    print(x[0].shape)