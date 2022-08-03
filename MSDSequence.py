from tensorflow.keras.utils import to_categorical, Sequence
import math
import os
import numpy as np

class MSDSequence(Sequence):
    def __init__(self, sample_list, data_path, batch_size, sample_size):
        self.sample_list = sample_list
        self.data_path = data_path
        self.batch_size = batch_size
        self.sample_size = sample_size

    def __len__(self):
        return math.ceil(self.sample_size / self.batch_size)
    
    def __load_data(self, image_file_path, label_file_path):
        with open(image_file_path, 'rb') as f:
            X = np.load(f)
        with open(label_file_path, 'rb') as f:
            y = np.load(f)

        return X,y

    def __getitem__(self, idx):
        batch_list = self.sample_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        
        for item in batch_list:
            X, y = self.__load_data( \
                os.path.join(self.data_path, item['image']), \
                os.path.join(self.data_path, item['label']), \
            )
            
            batch_x.append(X)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)