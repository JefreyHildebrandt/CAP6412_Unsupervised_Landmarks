import os
import sys


CELEBA = 'celeba'

def main(base_loc, dataset_type):
    from paone.datasets.celeba_dataset import CelebaDataset
    import paone.models.base_features as base_features_model
    dataset = None
    training_imgs_loc = None
    testing_imgs_loc = None 
    full_dataset_loc = os.path.join(base_loc, dataset_type)
    if(dataset_type == CELEBA):
        dataset = CelebaDataset(full_dataset_loc)
        training_imgs_loc = dataset.get_training_images_locs()
        testing_imgs_loc = dataset.get_testing_images_locs()
    #TEMP TEST FUNCTIONS
    # unprocessed_images = dataset.get_images(training_imgs_loc[0: 10])
    orig_imgs = dataset.pre_process_images(training_imgs_loc[0: 10])
    distort_imgs = dataset.get_view_point_changes(training_imgs_loc[0: 10])
    
    base_features_model.get_estimator(distort_imgs, orig_imgs)


if __name__ == "__main__":
    #get test images for celeba
    sys.path.append("/home/jeff/Documents/Projects/CAP6412_Advanced_CV_Project_One")
    BASE_LOC = '/home/jeff/Documents/Projects/data/datasets'
    main(BASE_LOC, CELEBA)