using Documenter, MDE_project

makedocs(
    sitename = "MDE for Multiscale Documentation",
    authors = "Jaroslav Borodavka",
    pages = [
        "Welcome" => "index.md",
        "Multiscale System and Homogenized Limit Pairs" => "multiscale_limit_pairs.md",
        "Invariant Densities" => "invariant_densities.md",
        "Cost Functionals for the MDE" => "MDE_functionals.md",
    ],
)