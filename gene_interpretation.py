import os
import argparse
import numpy as np
import dnnlib
import torch
import scvi
import anndata
import legacy
import tqdm

def _stylegan(label):
    network_pkl = "/nfs/turbo/umms-welchjd/hojaelee/stylegan2-ada-pytorch/patchseq_nuclei/00005-step3_filtered-cond-auto1-noaug/network-snapshot-000400.pkl"
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device) # type: ignore

    with dnnlib.util.open_url(network_pkl) as f:
        # G_mapping = legacy.load_network_pkl(f)["G_mapping"].to(device)
        # G_synthesis = legacy.load_network_pkl(f)["G_synthesis"].to(device)
        D = legacy.load_network_pkl(f)["D"].to(device)

    z = torch.from_numpy(np.random.randn(label.shape[0], G.z_dim)).to(device)
    img = G(z, label, truncation_psi=1, noise_mode="const") # (1, 3, 512, 512)
    logits = D(img, label)
    loss_Gmain = torch.nn.functional.softplus(-logits)
    loss_G = loss_Gmain.mean()

    return loss_G

def _scvi(x):
    scvi_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/patchseq/data/processed/gene_expression/patchseq_nuclei_scVI"
    adata = anndata.read_h5ad(os.path.join(scvi_path, "adata.h5ad"))
    model = scvi.model.SCVI.load(scvi_path, adata)

    x_ = x.cuda()
    x_ = torch.log(1 + x_)

    encoder_input = x_
    categorical_input = tuple()
    batch_index = torch.zeros([10, 1])
    batch_index += 2
    qz_m, qz_v, z = model.module.z_encoder(encoder_input, batch_index, *categorical_input)
    del model

    return z

def interpret_scvi(device, cell_source, model, adata, indices):
    batch_size = 8
    scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)
    del model

    for i, tensors in enumerate(scdl):
        print(f"Batch {i} out of {len(scdl)}...")
        if batch_size * (i+1) > len(indices):
            sub_index = indices[(batch_size * i):]
        else:
            sub_index = indices[(batch_size * i):(batch_size * (i+1))]

        tensors["X"].requires_grad = True
        z = _scvi(tensors["X"])
        # jacobian_latent = torch.autograd.functional.jacobian(_scvi, tensors["X"]) # (1, 10, 1, 2133)
        scvi_jacobian = batch_jacobian(_scvi, tensors["X"])

        save_path = f"./interpretation/{cell_source}_scvi_jacobian_batch{i}.npy"
        np.save(save_path, scvi_jacobian.detach().cpu().numpy())

        # stylegan
        stylegan_jacobian = torch.autograd.functional.jacobian(_stylegan, z)
        save_path = f"./interpretation/{cell_source}_stylegan2_jacobian_batch{i}.npy"
        np.save(save_path, stylegan_jacobian.detach().cpu().numpy())

        # cell indices
        sub_index_np = np.array(sub_index)
        save_path = f"./interpretation/{cell_source}_index_batch{i}.npy"
        np.save(save_path, sub_index_np)

def select_cells(adata, cell_source):
    obs = adata.obs
    obs = obs.reset_index()
    obs["cell_source"] = obs["cell_source"].astype(str)
    obs["celltype"] = obs["celltype"].astype(str)

    subset_df = obs.loc[(obs["cell_source"] == cell_source)]

    cell_indices = list(subset_df.index)

    return cell_indices

def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

def main(dataset, cell_source):
    device = torch.device('cuda')
    # os.makedirs(outdir, exist_ok=True)

    # Interpret scVI, S = (10, 2133)
    scvi_path = "/nfs/turbo/umms-welchjd/hojaelee/datasets/patchseq/data/processed/gene_expression/patchseq_nuclei_scVI"
    adata = anndata.read_h5ad(os.path.join(scvi_path, "adata.h5ad"))
    model = scvi.model.SCVI.load(scvi_path, adata)

    indices = select_cells(adata, cell_source)
    interpret_scvi(device, cell_source, model, adata, indices)

    """
    # Compute overall gradient w.r.t. genes
    gene_grad = torch.zeros(len(indices), 2133)
    for i in range(len(indices)):
        Si = S[i,:] # (2133, 10)
        Gi = G[i,:].T # (10)
        gradi = torch.matmul(Si.cuda(), Gi.cuda()) # (2133)

        gene_grad[i,:] = gradi.abs()

    np.save(f"./interpretation/{cell_source}_genegrad.npy", gene_grad.detach().cpu().numpy())
    """

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply closed form factorization")
    parser.add_argument("--dataset", default="patchseq_nuclei", type=str)
    parser.add_argument("--cellsource", default="tolias", type=str)
    args = parser.parse_args()

    dataset = args.dataset
    cell_source = args.cellsource
    main(dataset, cell_source)

#----------------------------------------------------------------------------
