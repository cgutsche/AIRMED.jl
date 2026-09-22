using Documenter
using AIRMED

# Replace USER and REPO with the GitHub repository that hosts this package.
# `remote` is stated explicitly so that the build does not depend on a
# configured git remote, and `deploydocs` pushes the rendered site to the
# gh-pages branch, from which GitHub Pages serves it.
const USER = "cgutsche"
const REPO = "AIRMED.jl"

DocMeta.setdocmeta!(AIRMED, :DocTestSetup, :(using AIRMED); recursive = true)

makedocs(;
    modules  = [AIRMED],
    authors  = "AIRMED Contributors",
    sitename = "AIRMED.jl",
    repo     = Documenter.Remotes.GitHub(USER, REPO),
    format   = Documenter.HTML(;
        canonical     = "https://$USER.github.io/$REPO",
        edit_link     = "main",
        prettyurls    = get(ENV, "CI", "false") == "true",
        collapselevel = 1,
    ),
    pages = [
        "Home"          => "index.md",
        "Workflow"      => "workflow.md",
        "API contracts" => "api.md",
        "Design notes"  => "design.md",
        "API reference" => "reference.md",
        "Internals"     => "internals.md",
    ],
    # Every docstring is included, the public ones through explicit @docs
    # blocks and the internal ones through @autodocs on the Internals page.
    checkdocs = :all,
)

deploydocs(;
    repo      = "github.com/$USER/$REPO.git",
    devbranch = "main",
    push_preview = true,
)
