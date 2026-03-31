import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
# module imports
from utils import text as text


class CelebADataset(Dataset):
    """
    Custom Dataset for loading CelebA face images
    
    partition: 0 for training, 1 for validation, 2 for test
    transform: whether or not to apply the cropping and resizing transformation to the images (default True, as this is what was used in the experiments)
    
    """
    def __init__(self, cfg, alphabet, partition=0, transform=None):
        # store config variables
        self.cfg = cfg
        self.alphabet = alphabet
        # image directory
        self.img_dir = os.path.join(cfg.dataset.dir_data, "img_align_celeba")
        # whether or not to apply the cropping and resizing transformation to the images (default True, as this is what was used in the experiments)
        self.transform = transform
        
        # load file with partition information (train/val/test)
        filename_partition = os.path.join(
            cfg.dataset.dir_data, "list_eval_partition.csv"
        )
        self.partition_path = filename_partition
        df_partition = pd.read_csv(filename_partition)        
        # load file with text information, file names and text modality 
        filename_text = os.path.join(
            cfg.dataset.dir_data,
            "list_attr_text_"
            + str(cfg.dataset.len_sequence).zfill(3)
            + "_"
            + str(cfg.dataset.random_text_ordering)
            + "_"
            + str(cfg.dataset.random_text_startindex)
            + "_celeba.csv",
        )
        self.txt_path = filename_text
        df_text = pd.read_csv(filename_text)
        df_text = df_text.loc[df_partition["partition"] == partition]
        
        # load file with the 40 attributes for each image
        filename_attributes = os.path.join(cfg.dataset.dir_data, "list_attr_celeba.csv")
        self.attrributes_path = filename_attributes
        df_attributes = pd.read_csv(filename_attributes)
        df_attributes = df_attributes.loc[df_partition["partition"] == partition]
        self.attributes = df_attributes # not strictly needed
        # the names of the 40 attributes used within classifiers e.g. "beard"
        self.label_names = list(df_attributes.columns)[1:] 
        
        ## store data for partition samples
        # text modality 
        self.y = df_text["text"].values 
        # original binary attributes as labels for classifiers
        self.labels = df_attributes.values
        # image file names
        self.img_names = df_text["image_id"].values 
        

    def __getitem__(self, index):
        with Image.open(os.path.join(self.img_dir, self.img_names[index])) as img:
            if self.transform is not None:
                # crop the original 218x178 image to 148x148, then resize to cfg.dataset.img_size x cfg.dataset.img_size (default 64x64)
                img = self.transform(img)
            # one-hot encode the text using the length of a sequence, the  alphabet and the text modality labels (the "text" column in the csv file)
            # returns a tensor of shape (length of sequence, length of alphabet)
            # X_ij = 1 if the i-th character in the sequence is the 
            # j-th character in the alphabet, 0 otherwise
            text_str = text.one_hot_encode(
                self.cfg.dataset.len_sequence, self.alphabet, self.y[index]
            )
            # extract the original labels (a binary vector of length 40 for the 40 attributes)
            label = torch.from_numpy((self.labels[index, 1:] > 0).astype(int)).float()
            sample = {"img": img, "text": text_str}
            return sample, label

    def __len__(self):
        return self.y.shape[0]

    def get_text_str(self, index):
        return self.y[index]

# class full_dataset_celebA(scMNC):
#     def __init__(self, cfg, training=False):
#         super().__init__(cfg.dataset.dir_data, cfg.model.seed,train=training)
#         self.data_loader = torch.utils.data.DataLoader(
#             self,
#             batch_size=3654 if training else 731,
#             shuffle=False,
#             num_workers=cfg.dataset.num_workers,
#             drop_last=False,
#         )
#         self.batch = next(iter(self.data_loader))
#         self.exp_data = self.batch[0]["exp"].numpy() # expression data
#         self.feat_data = self.batch[0]["feat"].numpy() # feature data
#         self.labels = self.batch[1].numpy() # tensor of labels
