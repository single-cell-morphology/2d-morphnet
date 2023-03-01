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
import tqdm
import random

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
@click.option('--cell_source', help='Tolias, Allen, or Nuclei', required=True, type=str)
@click.option('--celltype', help='IT, Sst, Pvalb, Vip, Lamp5, Sncg', required=True, type=str)
@click.option('--num_imgs', type=int, help='Number of cells to sample per celltype')
@click.option('--from_train', type=bool, required=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    dataset: Optional[str],
    cell_source: str,
    celltype: str,
    num_imgs: int,
    from_train: bool
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
    python generate.py --outdir=generated/patchseq_cond --seeds=0 --class=1 --dataset="patchseq" --cell_source="tolias" --celltype="Pvalb" --num_imgs=10 --from_train=False --network=./patchseq/00019-step3_filtered-cond-auto1-noaug/network-snapshot-000800.pkl

    \b
    # Patchseq_Nuclei
    python generate.py --outdir=generated/patchseq_nuclei_cond --seeds=0 --class=1 --dataset="patchseq_nuclei" --cell_source="patchseq" --celltype="Pvalb" --num_imgs=10 --from_train=False --network=./patchseq_nuclei/00005-step3_filtered-cond-auto1-noaug/network-snapshot-000400.pkl
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

    random.seed(seeds[0])
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0: # model is conditional
        if dataset == "patchseq":
            scvi_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/patchseq/data/processed/gene_expression/patchseq_scVI"
        elif dataset == "patchseq_nuclei":
            scvi_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/patchseq/data/processed/gene_expression/patchseq_nuclei_scVI"
        elif dataset == "merscope":
            scvi_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/merscope/processed/scVI/merscope-baseline"

        scvi_model = scvi.model.SCVI.load(scvi_path)
        adata = anndata.read_h5ad(os.path.join(scvi_path, "adata.h5ad"))
        adata.obs = adata.obs.reset_index()

        # Determine which images to produce for conditional generation
        cell_indices = select_cells(adata, cell_source, celltype, num_imgs, from_train)
        label = scvi_model.get_latent_representation(adata, indices=cell_indices, give_mean=True) # (107411, 10)
        label = torch.from_numpy(label).to(device)
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating images for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        for i, cell_index in enumerate(cell_indices):
            cell_id = adata.obs.at[cell_index, "index"]
            img = G(z, torch.unsqueeze(label[i], 0), truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{cell_source}_{celltype.split(" ")[1]}_{cell_id}_train{int(from_train)}_seed{seed:04d}.png')

def select_cells(adata, cell_source, celltype, num_imgs, from_train=False):
    obs = adata.obs
    if cell_source == "patchseq":
        obs["cell_source"] = obs["cell_source"].astype(str)
        obs["celltype"] = obs["celltype"].astype(str)
        subset_df = obs.loc[((obs["cell_source"] == "tolias") | (obs["cell_source"] == "allen")) & (obs["celltype"] == celltype) & (obs["use_train"]==int(from_train))]
    elif cell_source == "merscope":
        subset_df = obs.loc[obs["neuron_or_not"] == int(celltype)]
    else:
        subset_df = obs.loc[(obs["cell_source"] == cell_source) & (obs["celltype"] == celltype) & (obs["use_train"]==int(from_train))]

    if num_imgs == -1:
        cell_indices = list(subset_df.index)
    else:
        cell_indices = random.choices(list(subset_df.index), k=num_imgs)

    return cell_indices

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
