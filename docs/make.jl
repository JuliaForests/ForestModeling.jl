using ForestModeling
using Documenter

DocMeta.setdocmeta!(ForestModeling, :DocTestSetup, :(using ForestModeling); recursive=true)

makedocs(;
    modules=[ForestModeling],
    authors="Marcos Daniel da Silva <marcosdasilva@5a.tec.br> and contributors",
    sitename="ForestModeling.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaForests.github.io/ForestModeling.jl",
        edit_link="master",
        assets=String[],
        # index.md now carries a full worked example for every exported function
        # plus the @autodocs API reference on one page, past the default 100 KiB
        # size warning (still comfortably under the 200 KiB error threshold).
        size_threshold_warn=200 * 1024,
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaForests/ForestModeling.jl",
    devbranch="master",
)
