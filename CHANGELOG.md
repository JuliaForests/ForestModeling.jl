## [Unreleased]

The package was rewritten from the start. It no longer searches catalogs of transformations of the
response nor corrects the bias of their back-transformation: every model is fitted on the original
scale of the response, with the established Julia package of each method.

- `fitglm`: generalized linear models (family + link) by GLM.jl, the ordinary linear regression by `lm`.
- `fitnls`: nonlinear regression with the variance growing with a power of the mean, by
  Levenberg–Marquardt (LsqFit.jl) and maximum likelihood (Optim.jl).
- `fitmixed`: linear mixed models by MixedModels.jl.
- `fitbygroup`: one fit per category, in parallel threads.
- `predict`: on the original scale, in the unit of the response, with confidence and prediction
  intervals (delta method for the nonlinear regression).
- `calibrate`: the random effects of new plots from a few measured trees, for the linear mixed models.
- `blockcv` and `metrics`: cross-validation by plot or stand; RMSE, bias and MAPE.
- `robustvcov`: sandwich covariance.
- The equations `chapmanrichards`, `weibull`, `prodan`, `curtis`, `schumacherhall`, `spurr` and
  `power`; the height of reference of the height–diameter curves is always given (`href`).
- The columns may be `Unitful` quantities; ForestFoundations is re-exported.
- The models are shown as their fitted equation on the original scale, in one line in the cells of
  a table (`v = exp(-12.02 + 1.988 * log(d) + 1.563 * log(h))`), and with the table of the
  coefficients, the unit of the response and `n` when shown alone.
- Taper and assortments, revised from the drafts of ForestModeling and ForestMensuration: the
  taper equations `kozak1969` (now in its published form, ``(d/D)^2``), `schoepfer1966`,
  `ormerod1973`, `demaerschalk1972`, `maxburkhart1976` and `kozak1988` (now in their published
  forms), `kozak2004` and `bi2000` (with the height of reference given, never 1.3 m fixed), plain
  functions fitted by `fitnls`; `taperdiameter`, `taperheight`, `taperedvolume` and
  `logassortment` apply any of them, with units, and `logassortment` returns a row per product
  (`name`, `n`, `v`). The forms of Matte (1949), Johnson (1911) and Clutter (1980) of the drafts
  were not kept: their published forms could not be confirmed. The stem profiles of 50 trees are
  the example data, a local file (`data/cubage.csv`) not in the repository; without it the tests
  of the taper fits and of the assortments are skipped.
- Checked against the R package timbeR (2.0.1) with the same coefficients, on the 50 trees: the
  diameters, the volumes, the heights of a diameter and the logs of each product of Schöepfer
  (`poly5_*`) and Bi (2000) (`bi_*`) agree (diameters to 1e-15, volumes to 1e-5, heights to 5e-4,
  the same logs once the loss misread by `bi_logs` is reproduced); `fitnls` gives the coefficients
  of `minpack.lm::nlsLM`. Kozak (2004) differs: timbeR (and fptools) use
  ``X = (1 - z^{1/4})/(1 - p^{1/3})`` with `p` estimated, not the published model 02
  (``1/3`` in both and ``p = 1.3/H``) of `kozak2004`.
- `fitnls` with `variance=:constant` no longer fails when the mean is zero somewhere (the tip of a
  stem).
- The volume equations of Honer (1965), `honer`, and of Takata (1958), `takata`.
- `graybilltest`: the F test of Graybill (1976) that the predictions equal the observations.
- `identitytest`: the test of the identity of models (Regazzi), by the likelihood ratio of the
  models fitted to each group against the one fitted to all of them; `loglikelihood` and `dof` of
  the models of the package.
- `logassortment(...; fixedlength=true)`: only logs of the longest length, as the Python package
  fptools cuts them.
- The tutorial shows the classic equations that are formulas of `fitglm` (Stoffels, Henricksen,
  parabolic, Spurr, Stoate, Näslund, Meyer); the page of taper shows the commercial volume up to a
  top diameter and the stem of the mean tree.
- The tutorial and the tests use the example data of ForestMensuration.jl (the heights of 55 trees
  in four plots, the volumes of 91 trees and the biomass of 46 trees) instead of simulated data.
