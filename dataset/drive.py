
"""This is the file for the DriveDataset subclass"""

import os
import numpy as np
from skimage import io as skio
import cv2
from PIL import Image
from utilities.image_preprocessing import preprocessing


from dataset.dataset_w_masks import DatasetWMasks
from network.drive import DriveNetwork

class DriveDataset(DatasetWMasks):


    def __init__(self, batch_size=1, WRK_DIR_PATH='./drive', TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd=True,
                 cv_train_inds = None, cv_test_inds = None, he_flag=False,clahe_flag=False,normalized_flag=False,gamma_flag=False):
        self.he_flag=he_flag
        self.clahe_flag=clahe_flag
        self.normalized_flag=normalized_flag
        self.gamma_flag=gamma_flag
        super(DriveDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                           TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                           cv_test_inds=cv_test_inds)

        self.train_images, self.train_masks, self.train_targets = self.train_data
        self.test_images, self.test_masks, self.test_targets = self.test_data
        

    def get_images_from_file(self, DIR_PATH, file_indices=None):

        images = []
        masks = []
        targets = []

        IMAGES_DIR_PATH = os.path.join(DIR_PATH, self.IMAGES_DIR)
        MASKS_DIR_PATH = os.path.join(DIR_PATH, self.MASKS_DIR)
        TARGETS_DIR_PATH = os.path.join(DIR_PATH, self.TARGETS_DIR)

        image_files = sorted(os.listdir(IMAGES_DIR_PATH))
        mask_files = sorted(os.listdir(MASKS_DIR_PATH))
        target_files = sorted(os.listdir(TARGETS_DIR_PATH))

        if file_indices is not None:
            image_files = [image_files[i] for i in file_indices]
            mask_files = [mask_files[i] for i in file_indices]
            target_files = [target_files[i] for i in file_indices]

        for image_file,mask_file,target_file in zip(image_files, mask_files, target_files):

            image_arr = cv2.imread(os.path.join(IMAGES_DIR_PATH,image_file), 1)
            image_arr = image_arr[:, :, 1]
            image_arr = preprocessing(image_arr,he_flag=self.he_flag, clahe_flag=self.clahe_flag, normalized_flag=self.normalized_flag, gamma_flag=self.gamma_flag)

            top_pad = int((DriveNetwork.FIT_IMAGE_HEIGHT - DriveNetwork.IMAGE_HEIGHT) / 2)
            bot_pad = (DriveNetwork.FIT_IMAGE_HEIGHT - DriveNetwork.IMAGE_HEIGHT) - top_pad
            left_pad = int((DriveNetwork.FIT_IMAGE_WIDTH - DriveNetwork.IMAGE_WIDTH) / 2)
            right_pad = (DriveNetwork.FIT_IMAGE_WIDTH - DriveNetwork.IMAGE_WIDTH) - left_pad
>>>>>>> hypertest commit 2

    @property

    def network_cls(self):
        return DriveNetwork


    def test_set(self):
        return np.array(self.test_images), np.array(self.test_masks), np.array(self.test_targets)

