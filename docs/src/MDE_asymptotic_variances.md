# Asymptotic Variances of the MDE

## Introduction
In the multiscale overdamped Langevin drift parameter estimation problem, cf. section 4 of the article XXX, the MDE is asymptotically normal under the true parameter 
``\vartheta_0 \in \Theta_0`` as ``\epsilon \rightarrow 0`` with 
```math
\begin{aligned}
    \sqrt{T_\epsilon} \left(\hat{\vartheta}_{T_\epsilon}(X_\epsilon) - \vartheta_0 \right) \overset{\mathcal{D}_{\psi(\vartheta_0)}}{\longrightarrow} J(\vartheta_0)^{-1} \mathcal{N}_1(0, \tau^2(\vartheta_0)), \quad \text{as } \epsilon \rightarrow 0.
\end{aligned}
```
The following functions calculate the asymptotic variance figuring above.

## Functions

```@docs
Σ_∞_QdP
Σ_∞_QrP
```

## Index

```@index
Pages = ["MDE_asymptotic_variances.md"]
```
