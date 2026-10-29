using ForestModeling
using Documenter

DocMeta.setdocmeta!(ForestModeling, :DocTestSetup, :(using ForestModeling); recursive=true)

makedocs(;
  modules=[ForestModeling],
  authors="Marcos Daniel da Silva <marcosdasilva@5a.tec.br> and contributors",
  sitename="ForestModeling.jl",
  format=Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaForests.github.io/ForestModeling.jl",
    edit_link="master",
    assets=String[],
  ),
  checkdocs=:exports,
  pages=[
    "ForestModeling Package" => "forestmodeling.md",
    "Getting Started" => "tutorial.md",
    "Taper and Assortments" => "assortment.md",
    "API Reference" => "reference.md",
    "Index" => "index.md",
  ],
)

deploydocs(;
  repo="github.com/JuliaForests/ForestModeling.jl",
  devbranch="master",
)
