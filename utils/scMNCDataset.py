import os
import json
import io
import pickle
from collections import Counter, OrderedDict, defaultdict
from sklearn import preprocessing

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# code based on that from https://github.com/daifengwanglab/JAMIE/blob/main/examples/notebooks/scMNC-Visual-Cortical.ipynb 
# data from https://github.com/daifengwanglab/scMNC/tree/main/mouse_motor_cortex

class scMNC(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, dir_data):
        """
        Args:
            unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                correspond by index. Therefore the numbers of samples of all datapaths should match.
        """
        super().__init__()
        self.dir_data = dir_data
        self.label_names = "type"
        # save all paths to individual files
        self.file_paths = {dp: [] for dp in range(self.num_modalities)}
        for dp in range(self.num_modalities):
            files = glob.glob(os.path.join(self.dir_data, "m" + str(dp), "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            print(num_files, len(files))
            assert len(files) == num_files
        self.num_files = num_files

    @staticmethod
    def _create_scmnc_dataset(
        dir_data,
        savepath
    ):
        """Structure the scMNC Dataset with labels under 'savepath'.

        Args:
            dir_data (str): path to directory that contains the scMNC data files.
            savepath (str): path to directory that the dataset will be written to. Will be created if it does not
                exist.
        """

        # load MNIST data
        data1 = pd.read_csv(os.path.join(dir_data, "geneExp_filtered.csv"))
        data2 = pd.read_csv(os.path.join(dir_data, "efeature_filtered.csv"))
        sample_names1 = data1.columns[1:]
        sample_names2 = np.array(data2)[:, 0]
        feature_names1 = data1.iloc[:,0]
        feature_names2 = data2.columns[3:]
        assert (sample_names1 == sample_names2).all()
        data1 = np.transpose(np.array(data1)[:, 1:])
        data2 = np.array(data2)[:, 3:]
        meta = pd.read_csv(
          os.path.join(dir_data, "20200711_patchseq_metadata_mouse.csv")
          )
        meta_names = np.array(meta.columns)
        meta_sid = np.argwhere(meta_names == 'transcriptomics_sample_id')[0][0]
        meta_ttype = np.argwhere(meta_names == 't_type')[0][0]
        meta = np.array(meta)
        meta_idx = [
          np.argwhere(meta[:, meta_sid] == sample_names1[i])[0][0] for i in     range(sample_names1.shape[0])
          ]
        type1 = np.array(
          [x.split(' ')[0] for x in meta[meta_idx, meta_ttype]]
          )

        features = [np.array(feature_names1), np.array(feature_names2)]
        feature_dict = {
          'upstroke_downstroke_ratio_short_square':
            'up-downstroke_ratio_short_square',
            'upstroke_downstroke_ratio_long_square':
              'up-downstroke_ratio_long_square'
              }
        # Preprocessing
        data1 = preprocessing.scale(data1, axis=0)
        data2 = preprocessing.scale(data2, axis=0)
        data1[np.isnan(data1)] = 0  # Replace NaN with average
        data2[np.isnan(data2)] = 0
        dataset = [data1, data2]

        # Replace NULL feature names
        for i in range(len(features)):
            if features[i] is None:
                features[i] = np.array([f'Feature {i}' for i in range(dataset[i].shape[1])])
        
                # save labels and data
        os.makedirs(savepath, exist_ok=True)
        with open(os.path.join(savepath, "labels.pkl"), "wb") as f:
            pickle.dump(type1, f)


    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        files = [self.file_paths[dp][index] for dp in range(self.num_modalities)]
        labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]
        images = [Image.open(files[m]) for m in range(self.num_modalities)]

        images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        
        return (
            images_dict,
            labels[0],
        )  # NOTE: for scMNC, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-modalities", type=int, default=5)
    parser.add_argument("--savepath-train", type=str, required=True)
    parser.add_argument("--savepath-test", type=str, required=True)
    parser.add_argument("--backgroundimagepath", type=str, required=True)
    parser.add_argument("--rotate-mnist", default=False, action="store_true")
    parser.add_argument("--translate-mnist", default=False, action="store_true")
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # create dataset
    scMNC._create_scmnc_dataset(
        args.dir_data,
        args.savepath_train
    )
    print("Done.")