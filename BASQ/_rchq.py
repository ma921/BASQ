import torch
import numpy as np
import gurobipy as gp


def recombination(
    pts_rec,         # random samples for recombination
    pts_nys,         # number of samples used for approximating kernel with Nystrom method
    num_pts,         # number of samples finally returned
    kernel,          # kernel
    device,          # device
    ratio=2,           # ratio in recombination: if set to be None, it runs a full LP
    init_weights=None,  # initial weights of the sample for recombination
    calc_obj=None,   # a function for calculating an additional objective function
):
    """
    Args:
        - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
        - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
        - num_pts: int, number of samples finally returned. In BASQ context, this is equivalent to batch size
        - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)
        - device: torch.device, cpu or cuda
        - init_weights: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        - calc_obj: a function that returns a Tensor of objective values for all the input points

    Returns:
        - x: torch.tensor, the sparcified samples from pts_rec. The number of samples are determined by self.batch_size
        - w: torch.tensor, the positive weights for kernel quadrature as discretised summation.
    """
    return rc_kernel_svd(pts_rec, pts_nys, num_pts, kernel, device, ratio, mu=init_weights, calc_obj=calc_obj)


def ker_svd_sparsify(pt, s, kernel, device):
    _U, S, _ = torch.svd_lowrank(kernel(pt, pt), q=s)
    U = -1 * _U.T  # Hermitian
    return S, U


def rc_kernel_svd(samp, pt, s, kernel, device, ratio, mu=None, calc_obj=None):
    # Nystrom method
    _, U = ker_svd_sparsify(pt, s - 1, kernel, device)
    w_star, idx_star = Mod_Tchernychova_Lyons(
        samp, U, pt, kernel, device, ratio, mu=mu, calc_obj=calc_obj
    )
    return idx_star, w_star


def Mod_Tchernychova_Lyons(samp, U_svd, pt_nys, kernel, device, ratio=2, mu=None, calc_obj=None, DEBUG=False):
    """
    This function is a modified Tcherynychova_Lyons from
    https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py
    """
    N = len(samp)
    n, length = U_svd.shape
    if ratio is None:
        number_of_sets = N
    else:
        number_of_sets = min(N, max(ratio, 2) * (n + 1))

    # obj = torch.zeros(N).to(device)
    if mu is None:
        mu = torch.ones(N).to(device) / N

    idx_story = torch.arange(N).to(device)
    idx_story = idx_story[mu != 0]
    remaining_points = len(idx_story)

    use_obj = False if calc_obj is None else True
    if use_obj:
        obj = calc_obj(samp)

    while True:
        if remaining_points <= n + 1:
            idx_star = torch.arange(len(mu))[mu > 0].to(device)
            w_star = mu[idx_star]
            return w_star, idx_star

        elif n + 1 < remaining_points <= number_of_sets:
            X_mat = U_svd @ kernel(pt_nys, samp[idx_story])
            if use_obj:
                X_mat = torch.cat((X_mat, torch.reshape(
                    obj[idx_story], (1, -1))), 0)
                #X_mat_raw = torch.clone(X_mat[:-1])
            # w_star, idx_star, x_star, _, ERR, _, _ = Tchernychova_Lyons_CAR(
            #     X_mat.T, torch.clone(mu[idx_story]), device, DEBUG)
            w_star, idx_star = LP(X_mat, torch.clone(
                mu[idx_story]), device, use_obj=use_obj)

            idx_story = idx_story[idx_star]
            mu[:] = 0.
            mu[idx_story] = w_star
            idx_star = idx_story
            w_star = mu[mu > 0]

            return w_star, idx_star

        number_of_el = int(remaining_points / number_of_sets)

        idx = idx_story[:number_of_el *
                        number_of_sets].reshape(number_of_el, -1)
        X_for_nys = torch.zeros((length, number_of_sets)).to(device)
        X_for_obj = torch.zeros((1, number_of_sets)).to(device)
        for i in range(number_of_el):
            idx_tmp_i = idx_story[i * number_of_sets:(i + 1) * number_of_sets]
            X_for_nys += torch.multiply(
                kernel(pt_nys, samp[idx_tmp_i]),
                mu[idx_tmp_i].unsqueeze(0)
            )
            if use_obj:
                X_for_obj += torch.multiply(
                    torch.reshape(obj[idx_tmp_i], (1, -1)), mu[idx_tmp_i].unsqueeze(0))

        X_tmp_tr = U_svd @ X_for_nys
        if use_obj:
            X_tmp_tr = torch.cat((X_tmp_tr, X_for_obj), 0)
        X_tmp = X_tmp_tr.T
        tot_weights = torch.sum(mu[idx], 0).to(device)
        idx_last_part = idx_story[number_of_el * number_of_sets:]

        if len(idx_last_part):
            X_mat = U_svd @ kernel(pt_nys, samp[idx_last_part])
            if use_obj:
                X_mat = torch.cat((X_mat, torch.reshape(
                    obj[idx_last_part], (1, -1))), 0)
            X_tmp[-1] += torch.multiply(
                X_mat.T,
                mu[idx_last_part].unsqueeze(1)
            ).sum(axis=0)
            tot_weights[-1] += torch.sum(mu[idx_last_part], 0)

        X_tmp = torch.divide(X_tmp, tot_weights.unsqueeze(0).T)

        # # sparsify for the case use_obj is True
        # if use_obj:
        #     X_tmp_raw = torch.clone(X_tmp[:, :n])
        #     obj_raw = X_tmp[:, -1:].reshape(-1)

        # w_star, idx_star, _, _, ERR, _, _ = Tchernychova_Lyons_CAR(
        #     X_tmp, torch.clone(tot_weights), device
        # )
        w_star, idx_star = LP(X_tmp.T, torch.clone(
            tot_weights), device, use_obj=use_obj)

        idx_tomaintain = idx[:, idx_star].reshape(-1)
        idx_tocancel = torch.ones(idx.shape[1]).to(torch.bool).to(device)
        idx_tocancel[idx_star] = 0
        idx_tocancel = idx[:, idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = torch.multiply(mu[idx[:, idx_star]], w_star)
        mu_tmp = torch.divide(mu_tmp, tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)

        idx_tmp = idx_star == number_of_sets - 1
        idx_tmp = torch.arange(len(idx_tmp))[idx_tmp != 0].to(device)
        # if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp) > 0:
            mu_tmp = torch.multiply(mu[idx_last_part], w_star[idx_tmp])
            mu_tmp = torch.divide(mu_tmp, tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp
            idx_tomaintain = torch.cat([idx_tomaintain, idx_last_part])
        else:
            idx_tocancel = torch.cat([idx_tocancel, idx_last_part])
            mu[idx_last_part] = 0.

        idx_story = torch.clone(idx_tomaintain)
        remaining_points = len(idx_story)


def LP(K_, mu_, device, ep=1e-5, er=1e-5, use_obj=True):
    # We solve an LP of the form
    # maximize c^T @ x
    # subject to |K @ x - K @ mu| <= |K| @ er
    #            x >= 0, |1^T @ x - 1| <= ep
    #
    # 'er' represents a relative L^1 error at each test function,
    #   either a scalar or a vector with length m - 1 (= # of test functions)
    # 'ep' represents an acceptable error of the sum of x from 1, a scalar
    #
    # If you want to integrate test functions "exactly",
    # we recommend using er = ep = 1e-5 or similar, depending on the precision you use

    K = K_.cpu().detach().numpy()
    mu = mu_.cpu().detach().numpy()
    m, n = K.shape
    mu = mu.reshape((n, 1))
    B = K @ mu
    B_er = (np.absolute(K[m-1, :]) @ mu) * er
    dum = np.ones((1, n))
    model = gp.Model("test")
    model.Params.LogToConsole = 0
    x = model.addMVar(n)
    model.update()

    if(use_obj == True):
        model.setObjective(- K[m - 1, :] @ x)
        # To maximize K[m-1,:] @ x
    else:
        model.setObjective(0)

    for i in range(m - 1):
        model.addConstr(K[i, :] @ x >= B[i] - B_er[i])
        model.addConstr(K[i, :] @ x <= B[i] + B_er[i])
    model.addConstr(dum @ x >= dum @ mu - ep)
    model.addConstr(dum @ x <= dum @ mu + ep)
    model.addConstr(x >= 0)

    model.optimize()
    idx_star = []
    w_star = []
    if model.Status == gp.GRB.OPTIMAL:
        for i in range(n):
            if x[i].X > 0:
                idx_star.append(i)
                w_star.append(x[i].X)
    else:
        print("FAILED")
    w_star = torch.from_numpy(np.reshape(w_star, -1)).to(device).float()
    # we use torch.float here, but you can edit here (and possibly other placed) to make it of double precision
    idx_star = torch.from_numpy(np.reshape(
        idx_star, -1).astype(int)).to(device)
    return w_star, idx_star

# def Tchernychova_Lyons_CAR(X, mu, device, DEBUG=False):
#     """
#     This functions reduce X from N points to n+1.
#     This is taken from https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py
#     """
#     X = torch.cat([torch.ones(X.size(0)).unsqueeze(0).T.to(device), X], dim=1)
#     N, n = X.shape
#     U, Sigma, V = torch.linalg.svd(X.T)
#     U = torch.cat([U, torch.zeros((n, N - n)).to(device)], dim=1)
#     Sigma = torch.cat([Sigma, torch.zeros(N - n).to(device)])
#     Phi = V[-(N - n):, :].T
#     cancelled = torch.tensor([], dtype=int).to(device)

#     for _ in range(N - n):
#         lm = len(mu)
#         plis = Phi[:, 0] > 0
#         alpha = torch.zeros(lm).to(device)
#         alpha[plis] = mu[plis] / Phi[plis, 0]
#         idx = torch.arange(lm)[plis].to(device)
#         idx = idx[torch.argmin(alpha[plis])]

#         if len(cancelled) == 0:
#             cancelled = idx.unsqueeze(0)
#         else:
#             cancelled = torch.cat([cancelled, idx.unsqueeze(0)])
#         mu[:] = mu - alpha[idx] * Phi[:, 0]
#         mu[idx] = 0.

#         if DEBUG and (not torch.allclose(torch.sum(mu), 1.)):
#             # print("ERROR")
#             print("sum ", torch.sum(mu))

#         Phi_tmp = Phi[:, 0]
#         Phi = Phi[:, 1:]
#         Phi = Phi - torch.matmul(
#             Phi[idx].unsqueeze(1),
#             Phi_tmp.unsqueeze(1).T,
#         ).T / Phi_tmp[idx]
#         Phi[idx, :] = 0.

#     w_star = mu[mu > 0]
#     idx_star = torch.arange(N)[mu > 0].to(device)
#     return w_star, idx_star, torch.nan, torch.nan, 0., torch.nan, torch.nan
