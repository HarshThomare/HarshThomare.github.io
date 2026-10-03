---
title: "I could not backprop the residual through the U-Net"
date: 2026-01-27
draft: false
categories: ["Machine Learning", "Physics"]
tags: ["U-Net", "PINN", "DeepONet", "Navier-Stokes"]
summary: "A U-Net for a flow field has no coordinate input, so the Navier-Stokes residual has nothing honest to differentiate. A trunk net on (x, y, t) is where that derivative can live."
description: "How to put a PINN residual on a U-Net field model by differentiating a trunk net instead of the convolutions."
---

I was reading a paper that uses a U-Net for scientific machine learning on the Navier-Stokes equations: Ma, Zhang, Thuerey, Hu, and Haidn, ["Physics-driven Learning of the Steady Navier-Stokes Equations using Deep Convolutional Neural Networks"](https://arxiv.org/abs/2106.09301). They put the physics in the loss. The derivatives in that loss are finite differences on the grid.

I wanted to extend that with a PINN, meaning I wanted to differentiate the network with respect to the coordinates. The lecture that fixed the idea for me is Steve Brunton's overview, ["Physics Informed Machine Learning"](https://www.youtube.com/watch?v=JoFW2uSd3Uo). The PDE residual goes in the loss, and automatic differentiation produces the derivatives.

## Where the backprop stopped

I could not do that on the U-Net. A convolution does not take (x, y, t) as inputs. The coordinates are indices into a tensor. If I interpolate the output grid and differentiate the interpolant, I get a stencil wearing an autograd costume. It is the same finite difference with extra steps.

So I went looking for a construction where the coordinate is an actual input to something I can differentiate.

## Putting the U-Net in the branch

The construction I found is the physics-informed DeepONet, from [Wang, Wang and Perdikaris](https://arxiv.org/abs/2103.10974). Shuai Guo wrote a clear implementation walkthrough, ["Operator Learning via Physics-Informed DeepONet: Let's Implement It From Scratch"](https://medium.com/data-science/operator-learning-via-physics-informed-deeponet-lets-implement-it-from-scratch-6659f3179887). I am not re-implementing that article. I am changing the branch.

The equation is 2D incompressible Navier-Stokes in vorticity form:

```text
∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = (1/Re) (∂²ω/∂x² + ∂²ω/∂y²)

u = ∂ψ/∂y,  v = −∂ψ/∂x,  ∇²ψ = −ω
```

I use vorticity so the residual is one scalar equation.

In a standard DeepONet the branch is an MLP on sensor values. Here the U-Net is the branch. It reads ω, or a short stack of frames, and emits K coefficient maps B_k. The trunk is an MLP on (x, y, t) that emits K basis functions φ_k. At a query point I sample the maps at the nearest cell and take the dot product with the trunk basis:

```text
ω(x, y, t) = Σ_k B_k(x, y) φ_k(x, y, t)
```

The trunk takes the coordinates, so the derivative lives there: ∂ω/∂y = Σ_k B_k ∂φ_k/∂y. The coefficients B_k are constant with respect to the query as long as the sample location is detached. The convolutions never see a derivative with respect to x, y or t.

Here is the code, shortened:

```python
def vorticity(field, coords, unet, trunk):
    coeff = unet(field)                          # (B, K, H, W)
    xy = coords[..., :2].detach()                # sample location is a constant
    sampled = nearest_cell(coeff, xy)            # (B, N, K)
    basis = trunk(coords)                        # (B, N, K), coords require grad
    return (sampled * basis).sum(dim=-1)

def ns_residual(omega, coords, u, v, re):
    ones = torch.ones_like(omega)
    d1 = torch.autograd.grad(omega, coords, ones, create_graph=True)[0]
    w_x, w_y, w_t = d1.unbind(dim=-1)
    d2x = torch.autograd.grad(w_x, coords, torch.ones_like(w_x), create_graph=True)[0][..., 0]
    d2y = torch.autograd.grad(w_y, coords, torch.ones_like(w_y), create_graph=True)[0][..., 1]
    return w_t + u * w_x + v * w_y - (1.0 / re) * (d2x + d2y)
```

Two details matter here.

The first is `create_graph=True`. It has to stay on in every call. The loss is the squared residual, and training differentiates that loss with respect to the weights. The residual already contains second derivatives, so the backward pass is differentiating through derivatives. If the first graph is dropped, the second derivative has nothing to attach to.

The second is `xy.detach()`. Without it, the gather that picks the nearest cell would pass gradients back through the sample location, and the residual would turn into a finite difference between neighboring cells. With the detach, gradients still reach the U-Net through dL/dB_k. They do not arrive through a pixel Laplacian.

## Velocity

I have not picked how to get u and v. Option A is a second trunk head for the stream function ψ. Then u and v are trunk derivatives too, and ∇²ψ + ω = 0 becomes another residual in the loss. Option B is to solve the Poisson equation for ψ on the grid, outside autodiff, and stop the gradient on u and v. Option B uses less memory. It also means the convective term cannot teach the trunk.

I have not trained this pair, so what you have here is the shape of the backward pass, not a result.
