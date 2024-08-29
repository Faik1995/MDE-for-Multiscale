# Optimization Task of the MDE

## Introduction

The MDE is based on the following minimization task
```math
\begin{aligned}
  \hat{\vartheta}_T(X_\epsilon) := \argmin_{\vartheta \in \Theta} \Delta_T(\vartheta, X_\epsilon),
\end{aligned}
```
where
```math
\begin{aligned}
    \Delta_T(\vartheta, X_\epsilon) := \int_{\R^d} \left| \frac1T \int_0^T \exp\left(i u^\top X_\epsilon(t)\right) \, dt - \int_{\R^d} \exp\left(i u^\top x\right) \mu(\vartheta, x) \, dx \right|^2 \varphi(u) \, du.
\end{aligned}
```
``\vartheta \in \Theta \subset \mathbb{R}^p`` is the parameter to be estimated and the data ``X_\epsilon`` comes in form of a time series from a multiscale process, cf. [Multiscale System and Homogenized Limit Pairs](@ref).

The specific implementation of the MDE depends highly on the chosen limit model, i.e. the homogenized limit process ``X`` to which the multiscale process ``X_\epsilon`` converges weakly as ``\epsilon \rightarrow 0``. This limit model is characterized by the invariant density ``\mu`` of the limit process ``X``. Building an estimator on basis of the invariant density leads to parameter identification problems which is why the implementation demands a prior estimation parameter (either diffusion or drift parameter) as input. 

Another component of the MDE is the weight function ``\varphi`` which is, in all considered cases, chosen as a centered Gaussian density with covariance matrix ``\beta^2 I_d, \beta > 0``, because this particular choice simplifies the computational cost of the implementation.

## Functions

```@docs
MDE
```

## Index

```@index
Pages = ["MDE_optimizers.md"]
```

