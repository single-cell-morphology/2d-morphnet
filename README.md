# MorphGAN 

How to pull from `stylegan2-ada-pytorch` repo:

```
git fetch upstream
git merge upstream/main main
```

Tips:
1. ImageFolderDataset requires `np.uint8` (pixel values ranging from [0, 255]) datasets.
2. Check Python and GCC version
```
which python # should be using conda python
which python # gcc/8.2 module should be used, not /bin/gcc
```
3. Install scVI, then install torch==1.7.0 (stylegan2-ada used torch v1.7, but scVI will try to install a higher version)