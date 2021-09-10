# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import scvi
import anndata

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--dataset', help='Name of Dataset', required=True, type=str)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    dataset_name: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Patchseq
    python generate.py --outdir=out --seeds=0 --class=1 --dataset="patchseq_nuclei" \\
        --network=./patchseq_512px_run/00013-xy_dpi96_512px-cond-auto1-gamma10-noaug/network-snapshot-004200.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    import pdb
    pdb.set_trace()

    if dataset_name == "patchseq_nuclei":
        # Load scVI model to use latent gene expression as labels
        scvi_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/patchseq/data/processed/gene_expression/tolias_allen_nuclei"
        scvi_model = 
    """
    def get_cell_indices(self):
        cell_indices = []
        for i in self._raw_idx:
            image_name = self._image_fnames[i].split(".")[0]
            cell_id = "_".join(image_name.split("_")[:-1])
            cell_index = self.adata_index.loc[self.adata_index["index"] == cell_id].index.values
            cell_indices.append(cell_index[0])

        return cell_indices

    elif self._dataset_name == "patchseq_nuclei":
                base_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/patchseq_combined/processed"
                scvi_path = "scVI/patchseq"
                self.scvi_model = scvi.model.SCVI.load(os.path.join(base_path, scvi_path))
                self.adata = anndata.read_h5ad(os.path.join(base_path, scvi_path, "adata.h5ad"))
                self.adata_index = self.adata.obs.reset_index()
                self.cell_indices = self.get_cell_indices()

    def _load_raw_labels(self):
        # Modify this in order to sample from scVI
        if self._dataset_name == "cifar10":
            fname = 'dataset.json'
            if fname not in self._all_fnames:
                return None
            with self._open_file(fname) as f:
                labels = json.load(f)['labels']
            if labels is None:
                return None
            labels = dict(labels)
            labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
            labels = np.array(labels)
            labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
            return labels
        elif self._dataset_name == "patchseq_nuclei":
            all_latent = self.scvi_model.get_latent_representation(self.adata, indices=self.cell_indices, give_mean=True)
            all_latent = all_latent.squeeze()
            return all_latent
    """

    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0: # G.c_dim=10
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
