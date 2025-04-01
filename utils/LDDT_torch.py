"""lDDT protein distance score."""
import torch

def lddt(predicted_points,
                 true_points,args,
                 cutoff=20.,
                 per_residue=False):
    
    if predicted_points.shape[0] != true_points.shape[0]:
        min_len = min(predicted_points.shape[0], true_points.shape[0])
        predicted_points = predicted_points[:min_len,:]
        true_points = true_points[:min_len,:]
    dmat_true = torch.sqrt(1e-10 + torch.sum((true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

    dmat_predicted = torch.sqrt(1e-10 + torch.sum(
            (predicted_points[:, :, None] -
             predicted_points[:, None, :])**2, axis=-1))

    # Exclude self-interaction.
    dists_to_score = (
            torch.matmul((dmat_true < cutoff).float(), (1. - torch.eye(dmat_true.shape[1])).to(args.device))
            )

    # Shift unscored distances to be far away.
    dist_l1 = torch.abs(dmat_true - dmat_predicted)


    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).float()   +
                                    (dist_l1 < 1.0).float()   +
                                    (dist_l1 < 2.0).float()   +
                                    (dist_l1 < 4.0).float()  )
    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + torch.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, axis=reduce_axes))

    return score
