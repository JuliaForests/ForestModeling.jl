```@meta
CurrentModule = ForestModeling
```

# Reference

```@docs
ForestModeling
```

## Equations

```@docs
chapmanrichards
weibull
prodan
curtis
schumacherhall
spurr
honer
takata
power
```

## Fitting

```@docs
fitglm
fitnls
fitmixed
fitbygroup
UnitfulModel
NLSFit
```

## Prediction and calibration

```@docs
predict(::UnitfulModel, ::Any)
calibrate
```

## Validation

```@docs
blockcv
metrics
graybilltest
identitytest
robustvcov
```

## Taper equations

```@docs
kozak1969
schoepfer1966
ormerod1973
demaerschalk1972
maxburkhart1976
kozak1988
kozak2004
bi2000
```

## Taper application and assortments

```@docs
taperdiameter
taperheight
taperedvolume
logassortment
```
