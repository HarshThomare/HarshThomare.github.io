---
title: "Let the Trunk Net Carry the Derivatives"
date: 2026-09-28
draft: false
categories: ["Machine Learning", "Physics"]
tags: ["DeepONet", "U-Net", "Navier-Stokes", "automatic differentiation"]
summary: "A U-Net is a good encoder of a vorticity field and a bad place to ask for a spatial derivative. The trunk net already takes coordinates as inputs, so that is where the Navier-Stokes residual should be differentiated."
description: "Why backpropagating a Navier-Stokes residual through a trunk net is a better fit than finite differences on a U-Net grid, and what that design gives up."
---

A physics-informed network wants coordinates, because the residual is made of derivatives with respect to coordinates. A U-Net wants a grid, because a flow is local and multiscale, and a convolution is a good way to read that structure. I have not trained the pairing of the two, so what follows is an argument about where the derivatives should live, and not a result.

## A residual needs coordinates, and a flow needs neighbors

The equation I have in mind is the two-dimensional incompressible Navier-Stokes equation in vorticity form:

```text
∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = (1/Re) (∂²ω/∂x² + ∂²ω/∂y²)

u = ∂ψ/∂y,   v = −∂ψ/∂x,   ∇²ψ = −ω
```

Taking the curl of the momentum equation removes the pressure, and a stream function ψ gives a velocity field that is divergence-free by construction. If I predicted (u, v, p) directly I would need a penalty on the divergence and a way to deal with the pressure. I choose vorticity so that the residual I differentiate is a single scalar equation. That choice has a cost, which I come back to near the end.

The physics-informed DeepONet is the model that takes coordinates, and the convolutional U-Net is the model that takes a grid. The inductive bias that suits a flow is local and multiscale, with eddies at one scale sitting on eddies at another, and that is what the grid model captures. People usually pick one of the two representations and live with the weakness of the other.

The DeepONet of Lu, Jin, Pang, Zhang and Karniadakis learns a solution operator G(u)(y). A branch network encodes the input function u, sampled at a fixed set of sensors. A trunk network encodes the query coordinate y. The output is the dot product of the two embeddings, plus a bias in the usual unstacked form. The branch in that setting is an MLP that sees a flattened list of sensor values. Nothing in it knows that two sensors are neighbors. For a flow field that is the wrong place to be blind.

The alternative, when the object is a field on a grid, is a U-Net or a Fourier neural operator in the sense of Li et al. A U-Net maps a grid to a grid. It knows about neighbors and it does not know what a coordinate is, because coordinates are indices and not inputs. The usual way to penalize a PDE residual on such a model is finite differences on the output grid. That is cheap, it is only as accurate as the stencil, and it gives no value between nodes unless you interpolate. So the derivatives have nowhere honest to live.

## The composition

Wang, Wang and Perdikaris, in "Learning the solution operator of parametric partial differential equations with physics-informed DeepONets" (arXiv:2103.10974), make the DeepONet dot product differentiable and add the PDE residual as an extra loss. The reason it works is simple. The query coordinate is an input to the trunk, so automatic differentiation of the output with respect to that coordinate is ordinary reverse mode. The branch outputs do not depend on the query, so for an output G = Σ b_k t_k the derivative is ∂G/∂y = Σ b_k ∂t_k/∂y. For readers who want code and not the paper, Shuai Guo's article "Operator Learning via Physics-Informed DeepONet: Let's Implement It From Scratch" (Medium, Towards Data Science) is a clear walkthrough of that construction. I am not re-implementing it here. I am asking what changes if the branch is a bad fit for a flow field.

My proposal is to let the U-Net replace the branch as a spatial encoder. It reads ω, or a short stack of recent frames, and emits K coefficient maps B_k. The trunk stays an MLP on (x, y, t). If the target has energy at high wavenumbers the trunk can start with a Fourier feature layer, but that is optional, and it makes the second derivatives stiffer. I have not tuned it and I do not claim a setting. At a query point, I sample the coefficient maps at that location and take the dot product with the trunk basis φ_k, so the predicted vorticity is

```text
ω(x, y, t) = Σ_k B_k(x, y) φ_k(x, y, t)
```

<figure>
  <img src="/images/trunk-unet.svg" alt="A vorticity grid enters a U-Net and becomes coefficient maps. Coordinates enter a trunk MLP. Their product is the predicted vorticity. A dashed arrow shows reverse-mode derivatives returning through the trunk into the Navier-Stokes residual.">
  <figcaption>The residual needs derivatives with respect to (x, y, t). Those inputs enter only through the trunk. The U-Net is trained through the coefficients, not through a pixel stencil.</figcaption>
</figure>

## The residual's derivatives go through the trunk

```python
def vorticity(field, coords, unet, trunk):
    # field:  (B, 1, H, W), a grid. coords: (B, N, 3) = (x, y, t), requires grad.
    coeff = unet(field)                         # (B, K, H, W)
    xy = coords[..., :2].detach()               # fixed cell; no stencil in ∂/∂x
    sampled = nearest_cell(coeff, xy)            # (B, N, K)
    basis = trunk(coords)                        # (B, N, K)
    return (sampled * basis).sum(dim=-1)         # (B, N)


def nearest_cell(coeff, xy):
    # xy is in [0, 1] and already detached. Nearest cell, not a bilinear stencil.
    _, channels, height, width = coeff.shape
    col = (xy[..., 0] * (width - 1)).round().clamp(0, width - 1).long()
    row = (xy[..., 1] * (height - 1)).round().clamp(0, height - 1).long()
    flat = coeff.flatten(2)                      # (B, K, H*W)
    index = (row * width + col).unsqueeze(1).expand(-1, channels, -1)
    return flat.gather(2, index).transpose(1, 2)


def ns_residual(omega, coords, u, v, re):
    ones = torch.ones_like(omega)
    d1 = torch.autograd.grad(omega, coords, ones, create_graph=True)[0]
    w_x, w_y, w_t = d1.unbind(dim=-1)
    d2x = torch.autograd.grad(w_x, coords, torch.ones_like(w_x), create_graph=True)[0][..., 0]
    d2y = torch.autograd.grad(w_y, coords, torch.ones_like(w_y), create_graph=True)[0][..., 1]
    return w_t + u * w_x + v * w_y - (1.0 / re) * (d2x + d2y)
```

`nearest_cell` gathers coefficient channels at a fixed grid location. Because `xy` was detached, that location is a constant with respect to the query, and the gather is not a stencil the derivative can leak through. The arguments `u` and `v` have to be supplied by one of the two routes I describe below.

The `create_graph=True` flag is there for a reason. The first backward pass builds ∂ω/∂x, but the loss still has to be differentiated with respect to the network weights, and the residual contains second derivatives of ω. If the graph from the first pass is thrown away, the second derivative has nothing to differentiate. So the graph must survive.

The coefficients are independent of the query coordinate only if the sample location is a constant, which is what `xy.detach()` enforces. If bilinear sampling is differentiated with respect to the sample coordinate, part of ∂ω/∂x comes from the jump between neighboring coefficients. That is a finite difference wearing an autograd costume. It can dominate the trunk's contribution, and then I have gained nothing over the stencil I was trying to avoid. So the sample location is stopped, or the coefficients are read at the query's nearest cell and held constant with respect to (x, y). Gradients still reach the U-Net, through ∂L/∂B_k. They do not arrive through a pixel Laplacian.

## Where the velocity comes from

The choice of velocity decides whether the convective term is still a trunk derivative or a quantity imported from a grid solve with the gradient stopped, and there are two honest ways to make it.

The first is a second, smaller head that predicts the stream function ψ the same way. Then (u, v) are derivatives of ψ taken through the same trunk, and ∇²ψ + ω = 0 becomes an additional residual that is also a trunk derivative. The whole residual stays inside one differentiable operator. It is also the more expensive option.

The second is to solve a Poisson equation for ψ on the grid from the U-Net's ω, outside the autodiff, and stop the gradient on the resulting velocities. This is closer to a classical CFD step. It also blocks the velocity from teaching the trunk anything, because the convective term then receives a fixed velocity from a different mechanism than the one that produced ω.

The second route makes sense when the Poisson solver is already trusted and memory is short, and using it changes the model. I have not chosen between them based on a run, because there has not been one.

## What this buys

Collocation points can sit off the grid, which was the original DeepONet point, and the convolutional encoder still sees a dense field. The residual does not have to be evaluated at pixel centers.

The more practical gain is the size of the derivative tape. The residual needs second derivatives, so the backward pass is deeper than for a supervised loss. The second-derivative tape here is the tape of a small MLP and not the tape of every convolution in the U-Net. A U-Net unrolled over many timesteps, with double backpropagation through the convective term, is a memory problem before it is an accuracy problem. Querying the trunk at collocation points for a single shot, or over a short time horizon, keeps that tape bounded. A trunk hidden width on the order of the ones in the DeepONet papers is the obvious place to start.

There is a second, smaller point. The U-Net can still be trained with a supervised term on the grid, where finite differences are unnecessary because there is a target field to compare against. The physics loss and the data loss do not have to take their derivatives from the same representation.

## What it gives up

The costs are real, and some of them could sink the idea.

The trunk basis can be too smooth for a turbulent cascade. Widening it or adding Fourier features raises the cost of ∇², and it can make the gradient of the convective term explode, because u·∇ω is quadratic in the fields.

Stopping the gradient on the sampler means the model cannot move a sharp front by sliding the sample coordinate. The front has to appear in B_k or in φ_k. If neither can represent it, the hybrid will smear it.

Boundaries are not free. The trunk does not know the domain unless the loss says so, either with a boundary residual or with a coordinate parameterization that already satisfies the condition. Periodic boundaries and walls need separate treatment. The vorticity form also makes the wall or far-field vorticity the boundary object, which is sometimes less natural than setting u = 0.

And none of this removes the need for a sensible Re, a sensible nondimensionalization, or a collocation set that actually sees the layer one cares about.

Finally, I have not shown that the hybrid beats an FNO with a finite-difference residual. The reason to try it is local: the derivative operator and the translation-equivariant encoder want different inputs, and the dot product is a place where they can meet without forcing either of them to fake the other's job.

The U-Net reads the field, the trunk reads the coordinate, and the residual differentiates the trunk.
