
import torch



def test():
    x = torch.randn(2, 3)
    w = torch.randn(3, 4)
    y = torch.matmul(x, w)

    # calculate xt
    xt = x.T
    # calculate xtx
    xtx = torch.matmul(xt, x)
    # calculate xty
    xty = torch.matmul(xt, y)

    print(xtx @ w)
    print(xty)
    print(torch.allclose(xtx @ w, xty))

    # solve xtx @ w = xty

    w_star = torch.linalg.solve(xtx, xty)

    print(xtx @ w_star - xty)


def solve_w(x, y):
    # calculate xt
    xt = x.T
    # calculate xtx
    xtx = torch.matmul(xt, x)
    # calculate xty
    xty = torch.matmul(xt, y)

    w_star = torch.linalg.solve(xtx, xty)

    # check if the result is correct
    assert torch.allclose(xtx @ w_star, xty)
    assert torch.allclose(x @ w_star, y)
    print(x)
    print(y)
    print(w_star)

    return w_star


solve_w(torch.randn(2, 3), torch.randn(2, 4))
