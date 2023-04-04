# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.16 (default, Mar  2 2023, 03:21:46) 
# [GCC 11.2.0]
# Embedded file name: /home/jinyuanshi/workspace/sparsegpt/reconstruct_weight.py
# Compiled at: 2023-03-31 19:19:18
# Size of source mod 2**32: 5230 bytes
import math, torch
import torch.nn as nn

# def invert(H: torch.Tensor, ridge_regular: float=0.0001) -> torch.Tensor:
#     Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + ridge_regular * H.diag().mean() * torch.eye((H.shape[0]), device=(H.device))))
#     return Hinv.to(H.dtype)

def invert(H: torch.Tensor, ridge_regular=1e-4):
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + ridge_regular * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device)))
    except RuntimeError:
        return invert(H=H, ridge_regular=ridge_regular * 10)
    return Hinv

def reconstruct_best_weight(xtx: torch.Tensor, x:torch.Tensor, y:torch.Tensor, w:torch.Tensor, ridge_regular=1e-4):
    mse = nn.MSELoss()
    # for ridge_regular in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    for ridge_regular in [ridge_regular]:
        w_star = torch.matmul(torch.matmul(invert(xtx, ridge_regular=ridge_regular), x.T), y)
        error = mse(w, w_star)
        if error.item() < 1:
            # print(error.item())
            break
    return w_star


def reconstruct_weight(x: torch.Tensor, y: torch.Tensor, ridge_regular: float=0.01) -> torch.Tensor:
    org_dtype = x.dtype
    x = x.float()
    y = y.float()
    x = x.view(-1, x.shape[-1])
    xtx = torch.matmul(x.T, x)
    w_star = torch.matmul(torch.matmul(invert(xtx, ridge_regular=ridge_regular), x.T), y.view(-1, y.shape[-1]))
    w_star = w_star.to(org_dtype)
    return w_star.T


@torch.no_grad()
def solve_weight(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, ridge_regular: float=0.01) -> torch.Tensor:
    org_dtype = x.dtype
    x = x.float()
    y = y.float()
    x = x.view(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])
    weight = weight.T
    weight = weight.float()
    xt = x.T
    xtx = torch.matmul(xt, x)
    xty = torch.matmul(xt, y)
    original_error = torch.norm(x.cpu() @ weight.cpu() - y.cpu())
    num_iter = int(math.log10(ridge_regular) * -1)
    replaced = False
    if not (-1e-10 < xtx.det() < 1e-10 or torch.isnan(xtx.det())):
        print('try reconstruct weight without adding ridge_regular to xtx because xtx is invertible')
        # w = torch.linalg.lstsq(xtx, xty).solution
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(xtx + ridge_regular * xtx.diag().mean() * torch.eye((xtx.shape[0]), device=(xtx.device))))
        w = torch.matmul(torch.matmul(Hinv, x.T), y.view(-1, y.shape[-1]))
        error = torch.norm(xtx.cpu() @ w.cpu() - xty.cpu())
        new_original_error = torch.norm(x.cpu() @ w.cpu() - y.cpu())
        print('error between xtx @ w and xty: {}'.format(error))
        print('error between x @ w and y: {}'.format(new_original_error))
        if new_original_error < original_error:
            original_error = new_original_error
            weight = w
            replaced = True
    else:
        for i in range(num_iter):
            print('try to solve the xtx is not invertible problem by adding {} to xtx'.format(ridge_regular))
            # w = torch.linalg.lstsq(xtx + ridge_regular * torch.eye((xtx.shape[0]), device=(xtx.device)) * xtx.diag().mean(), xty).solution
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(xtx + ridge_regular * xtx.diag().mean() * torch.eye((xtx.shape[0]), device=(xtx.device))))
            w = torch.matmul(torch.matmul(Hinv, x.T), y.view(-1, y.shape[-1]))
            error = torch.norm(xtx.cpu() @ w.cpu() - xty.cpu())
            new_original_error = torch.norm(x.cpu() @ w.cpu() - y.cpu())
            print('error between xtx @ w and xty: {}'.format(error))
            print('error between x @ w and y: {}'.format(new_original_error))
            if new_original_error < original_error:
                original_error = new_original_error
                weight = w
                replaced = True
            ridge_regular *= 10
    if replaced:
        print('============================================')
        print('Reconstruct weight successfully')
        print('============================================')
    else:
        print('============================================')
        print('Reconstruct weight failed, use the original weight')
        print('============================================')
    weight = weight.T.to(org_dtype)
    return weight

@torch.no_grad()
def solve_weight_jw(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, ridge_regular: float=0.01, sparsity: float=0.5) -> torch.Tensor:
    # get the weight name in nn.linear
    org_weight_dtype = weight.dtype
    if torch.norm(nn.functional.linear(x, weight, bias) - y) > 1e-3:
        if bias is not None:
            y = y - bias
        y = y.view(-1, y.shape[-1]).float()
        x = x.view(-1, x.shape[-1]).float()
        xtx = torch.matmul(x.T, x)
        # weight = solve_weight(x, y, weight, ridge_regular=ridge_regular)
        weight = reconstruct_best_weight(xtx, x, y, w=weight.T, ridge_regular=1e-4).T
        del x
        del y
        torch.cuda.empty_cache()
    else:
        x = x.view(-1, x.shape[-1]).float()
        xtx = torch.matmul(x.T, x)

    weight = weight.float()
    w_sparse = prune_unstructured(weight, xtx, sparsity=sparsity)
    w_sparse = w_sparse.to(org_weight_dtype)
    return w_sparse, bias

    # org_dtype = x.dtype
    # weight = weight.T
    # if bias is not None:
    #     y = y - bias
    # x = x.view(-1, x.shape[-1])
    # x = x.float()
    # y = y.float()
    # weight = weight.float()
    # xtx = torch.matmul(x.T, x)
    # w_star = reconstruct_best_weight(xtx, x, y.view(-1, y.shape[-1]), w=weight, ridge_regular=1e-4)

    # w_sparse = prune_unstructured(w_star.T, xtx, sparsity=sparsity).T
    # # w_sparse = w_star

    # w_sparse = w_sparse.to(org_dtype)

    # w = w_sparse
    # # if bias is not None:
    # #     bias = w_sparse[-1, :]
    # #     w = w_sparse[:-1, :]
    # # else:
    # #     bias = None
    # #     w = w_sparse
    # weight = w.T

    # return weight, bias


def prune_unstructured(W: torch.Tensor, H: torch.Tensor, block_size=128, use_perm=True, sparsity=0.5):
    rows, columns = W.shape
    W = W.clone()
    H = H.clone()

    perm = None
    if use_perm:
        # random
        # perm = torch.randperm(W.shape[1])
        # greedy
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Hinv = invert(H)
        diag = torch.diagonal(Hinv)
        scores = torch.sum(((W ** 2) / diag), dim=0)
        perm = torch.argsort(-scores)
        W = W[:, perm]
        H = H[:, perm][perm, :]

    M = torch.zeros_like(W)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    Hinv = invert(H)
    for i1 in range(0, columns, block_size):
        if columns - i1 < block_size:
            break
        w = W[:, i1:]
        hinv = Hinv[i1:, i1:]
        diag = torch.diagonal(hinv)
        scores = ((w[:, :block_size] ** 2) / diag[:block_size])
        mask = (torch.argsort(torch.argsort(scores, dim=1), dim=1) < int(block_size * sparsity)).float()
        M[:, i1:i1+block_size] = mask

        for j in range(block_size):
            row = hinv[j, :]
            d = diag[j]
            delta = (row * w[:, j].unsqueeze(1) / d) * mask[:, j].unsqueeze(1)
            w[:, j:] -= delta[:, j:]
            hinv[j:, j:] -= (row.reshape(-1, 1) @ row.reshape(1, -1))[j:, j:] / d

    W[M.bool()] = 0.0
    if use_perm:
        W = W[:, torch.argsort(perm)]
    return W