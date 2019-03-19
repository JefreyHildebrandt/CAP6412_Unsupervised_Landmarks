import os
import tensorflow as tf
from PIL import Image
from .abstract_dataset_parse import AbstractDatasetParse
import numpy as np
import Augmentor
import re
import cv2
import numpy as np
 
class CelebaDataset(AbstractDatasetParse):
    def __init__(self, base_loc):
        self.testing_loc = os.path.join(base_loc, 'MAFL', 'testing.txt')
        self.training_loc = os.path.join(base_loc, 'MAFL', 'training.txt')
        self.image_loc = os.path.join(base_loc, 'Img', 'img_align_celeba_hq')
        self.image_distort_loc = os.path.join(self.image_loc, 'output')
        if not os.path.exists(self.image_distort_loc):
            p = Augmentor.Pipeline(self.image_loc)
            p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
            p.process()

    def get_training_images_locs(self):
        return self.get_generic_loc(self.training_loc)
    
    def get_testing_images_locs(self):
        return self.get_generic_loc(self.testing_loc)
    
    def get_generic_loc(self, loc):
        with open(loc, 'r') as img_file:
            imgs = img_file.readlines()
        return [os.path.join(self.image_loc, img).rstrip() for img in imgs]
    
    # This will a synthetic transformation
    def get_view_point_changes(self, img_locs):
        img_distort_list_all = os.listdir(self.image_distort_loc)
        img_distort_list = []
        img_distort_dict = dict()
        for img_distort in img_distort_list_all:
            orig_file_name = re.search('original_(.*).jpg_', img_distort).group(1) + '.jpg'
            img_distort_dict[orig_file_name] = img_distort
 
        for img in img_locs:
            img_file_name = os.path.basename(os.path.normpath(img))
            img_distort_list.append(os.path.join(self.image_distort_loc, img_distort_dict[img_file_name]))
        return (self.pre_process_images(img_distort_list))

    # def pre_process_images(self, img_locs):
    #     filename_queue = tf.train.string_input_producer(img_locs)
    #     # count_num_files = tf.size(img_locs)

    #     reader = tf.WholeFileReader()
    #     key, value = reader.read(filename_queue)
    #     img = tf.image.decode_jpeg(value)

    #     init_op = tf.global_variables_initializer()
    #     allImgs = []
    #     with tf.Session() as sess:
    #         sess.run(init_op)

    #         # Start populating the filename queue.

    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)

    #         for i in range(len(img_locs)): #length of your filename list
    #             image = img.eval() #here is your image Tensor :) 
    #             image = tf.image.resize_images(image, [128, 128])

    #             allImgs.append(image)

    #         coord.request_stop()
    #         coord.join(threads)
    #     return allImgs

    
    def pre_process_images(self, img_locs):
        all_imgs = []
        for img_loc in img_locs:
            img = cv2.imread(img_loc)
            img_res = cv2.resize(img, dsize=(128, 128))
            all_imgs.append(img_res)
        return np.float32(np.asarray(all_imgs))
