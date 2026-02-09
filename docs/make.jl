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
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaForests/ForestModeling.jl",
    devbranch="master",
)
