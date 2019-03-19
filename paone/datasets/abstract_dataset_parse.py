from abc import abstractmethod

class AbstractDatasetParse:
    def __init__(self, base_loc):
        pass
    
    @abstractmethod
    def get_training_images_locs(self):
        pass
    
    @abstractmethod 
    def get_testing_images_locs(self):
        pass
    
    # This will be either a synthetic transformation or a different view
    @abstractmethod 
    def get_view_point_changes(self, img_list):
        pass

    @abstractmethod 
    def pre_process_images(self, img_locs):
        pass