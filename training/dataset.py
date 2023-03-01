# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import contextlib
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import scvi
import anndata

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset_name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self.name = dataset_name
        self.raw_shape = list(raw_shape)
        self.use_labels = use_labels
        self.raw_labels = None
        self.label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self.raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self.raw_labels is None:
            self.raw_labels = self._load_raw_labels() if self.use_labels else None
            if self.raw_labels is None:
                self.raw_labels = np.zeros([self.raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self.raw_labels, np.ndarray)
            assert self.raw_labels.shape[0] == self.raw_shape[0]
            assert self.raw_labels.dtype in [np.float32, np.int64]
            if self.raw_labels.dtype == np.int64:
                assert self.raw_labels.ndim == 1
                assert np.all(self.raw_labels >= 0)
        return self.raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self.name

    @property
    def image_shape(self):
        return list(self.raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._dataset_name == "merscope" and self.use_labels:
            return [10]
        if self._dataset_name == "patchseq_nuclei" and self.use_labels:
            return [10]
        if self.label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self.label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self.label_shape = raw_labels.shape[1:]
        return list(self.label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        dataset_name,           # Name of Dataset, e.g. cifar
        img_path,                   # Path to directory or zip.
        scvi_path = None, # Path to scvi model if conditioning on gene expression
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.dataset_name = dataset_name
        self.img_path = img_path
        self.scvi_path = scvi_path
        self._zipfile = None

        if os.path.isdir(self.img_path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in os.walk(self.img_path) for fname in files}
        elif self._file_ext(self.img_path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self.img_path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        # Load scvi model for conditional information
        if self.scvi_path is not None:
            self.scvi_model = scvi.model.SCVI.load(self.scvi_path)
            self.adata = anndata.read_h5ad(os.path.join(self.scvi_path, "adata.h5ad"))
            self.adata_index = self.adata.obs.reset_index()
            self.cell_indices = self._get_cell_indices()

        super().__init__(dataset_name=name, raw_shape=raw_shape, dataset_name=dataset_name, **super_kwargs)

    def _get_cell_indices(self):
        cell_indices = []
        cell_ids = [self._image_fnames[i].split(".")[0] for i in self._raw_idx]

        self.adata_index["index"] = self.adata_index["index"].astype("category")
        self.adata_index["index"].cat.set_categories(cell_ids, inplace=True)
        cell_indices = self.adata_index.sort_values(by=["index"]).index.values

        return cell_indices

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.img_path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self.img_path, fname), 'rb')
        return self._get_zipfile().open(fname, 'r') if self._type == 'zip' else None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        id_xy = fname.split(".")[0]
        self.id = id_xy[:-3]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        if self.dataset_name == "merscope":
            image = np.uint8(image * 255)
        return image

    def _load_raw_labels(self):
        all_latent = self.scvi_model.get_latent_representation(adata, indices=self.cell_indices, give_mean=True)
        all_latent = all_latent.squeeze()

        return all_latent
