from training_batch import open_batch
import uuid
import os
from config import *
import json

class Dataset():
    """Dataset object which contains just the imgs paths for reduce the memory usage"""
    def __init__(self):
        self.batch = []
        self.settings = {}
        self.name = str(uuid.uuid4())

        self.active_batch = []

    def load_from_name(self, name):
        self.batch.extend(open_batch(name, return_path=True))

    def add(self,imgs_path):
        self.batch.append(imgs_path)
    
    def get(self, indexes = None):
        b_min = 0 
        b_max = -1
        if not(indexes is None):
            if not(type(indexes) is list) or len(indexes) < 2:
                b_max = indexes
            else:
                b_min = indexes[0]
                b_max = indexes[1]
        paths = np.array(self.batch)[b_min:b_max]
        batch = self.load(paths)
        return batch

    def load(self, paths):
        result = []
        for pair in paths:
            if not(type(pair) is list):
                result.append(np.load(pair))
                continue
            temp = []
            for p in pair:
                temp.append(np.load(p))
            result.append(temp)
        del self.active_batch
        self.active_batch = result
        return result

    def split(self, batch, cutoff=0.7):
        batch = np.array(self.batch)
        cut_index = int(cutoff * len(batch))

        b1 = Dataset()
        b1.batch = batch[:cut_index]
        b1.settings = self.settings #TODO
        b1.name = self.name + "_b1"
        b2 = Dataset()
        b2.batch = batch[cut_index:]
        b2.settings = self.settings #TODO
        b2.name = self.name + "_b2"

        return (b1, b2)
    
    def save(self,batch, name=None):      
        if not(os.path.exists(TRAINING_BATCH_FOLDER)):
            os.mkdir(TRAINING_BATCH_FOLDER)

        batch_uuid = self.name if name is None else name
        while os.path.exists(os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid))):
            self.name = str(uuid.uuid4())
            LOGGER.warn(f"Batch {batch_uuid} already exists, change to: {str(self.name)}")
            batch_uuid = self.name

        batch_path = os.path.join(TRAINING_BATCH_FOLDER,"batch_"+str(batch_uuid))
        os.mkdir(batch_path)

        with open(os.path.join(batch_path,'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=4)

        for i,img in enumerate(batch):
            cdens = img[0]
            vdens = img[1]
            np.save(os.path.join(batch_path,str(i)+"_cdens.npy"), cdens)
            np.save(os.path.join(batch_path,str(i)+"_vdens.npy"), vdens)

        LOGGER.log(f"batch with {len(batch)} images saved.")

        return True