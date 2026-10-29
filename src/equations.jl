"""
    chapmanrichards(d, θ; href)

Chapman–Richards height–diameter curve, `θ = (a, b, c)`:

```math
h = h_{ref} + a \\left(1 - e^{-b d}\\right)^{c}
```

`a` is the asymptote above `href`, the height at which the diameter is measured (it changes from
country to country, so it is always given).

# Example
```jldoctest
julia> chapmanrichards(20.0, [25.0, 0.08, 1.2]; href=1.3)
20.37265071142402
```
"""
chapmanrichards(d, θ; href) = href + θ[1] * (1 - exp(-θ[2] * d))^θ[3]

"""
    weibull(d, θ; href)

Weibull height–diameter curve, `θ = (a, b, c)`:

```math
h = h_{ref} + a \\left(1 - e^{-b d^{c}}\\right)
```

# Example
```jldoctest
julia> weibull(20.0, [25.0, 0.02, 1.3]; href=1.3)
16.941421549810553
```
"""
weibull(d, θ; href) = href + θ[1] * (1 - exp(-θ[2] * d^θ[3]))

"""
    prodan(d, θ; href)

Prodan height–diameter curve in its original form, not linearized, `θ = (β₀, β₁, β₂)`:

```math
h = h_{ref} + \\frac{d^2}{\\beta_0 + \\beta_1 d + \\beta_2 d^2}
```

# Example
```jldoctest
julia> prodan(20.0, [2.0, 0.8, 0.03]; href=1.3)
14.633333333333335
```
"""
prodan(d, θ; href) = href + d^2 / (θ[1] + θ[2] * d + θ[3] * d^2)

"""
    curtis(d, θ)

Curtis height–diameter curve, `θ = (β₀, β₁)`:

```math
h = e^{\\beta_0 + \\beta_1 / d}
```

# Example
```jldoctest
julia> curtis(20.0, [3.3, -8.0])
18.17414536944306
```
"""
curtis(d, θ) = exp(θ[1] + θ[2] / d)

"""
    schumacherhall(d, h, θ)

Schumacher–Hall volume equation, `θ = (β₀, β₁, β₂)`:

```math
v = \\beta_0 \\, d^{\\beta_1} h^{\\beta_2}
```

# Example
```jldoctest
julia> schumacherhall(20.0, 18.0, [5.0e-5, 1.9, 1.0])
0.2668084016785011
```
"""
schumacherhall(d, h, θ) = θ[1] * d^θ[2] * h^θ[3]

"""
    spurr(d, h, θ)

Spurr volume equation, of the combined variable ``d^2 h``, `θ = (β₀, β₁)`:

```math
v = \\beta_0 + \\beta_1 d^2 h
```

# Example
```jldoctest
julia> spurr(20.0, 18.0, [0.01, 3.5e-5])
0.262
```
"""
spurr(d, h, θ) = θ[1] + θ[2] * d^2 * h

"""
    power(x, θ)

Power (allometric) equation, `θ = (a, b)`; with `x = D` it is the biomass model ``M = aD^b``, with
`x = ρD²H` the compound one:

```math
y = a \\, x^{b}
```

# Example
```jldoctest
julia> power(20.0, [0.1, 2.4])
132.57816069359944
```
"""
power(x, θ) = θ[1] * x^θ[2]

"""
    honer(d, h, θ)

Honer (1965) volume equation, `θ = (β₀, β₁)`:

```math
v = \\frac{d^2}{\\beta_0 + \\beta_1 / h}
```

# Example
```jldoctest
julia> honer(20.0, 18.0, [400.0, 20000.0])
0.2647058823529412
```
"""
honer(d, h, θ) = d^2 / (θ[1] + θ[2] / h)

"""
    takata(d, h, θ)

Takata (1958) volume equation, `θ = (β₀, β₁)`:

```math
v = \\frac{d^2 h}{\\beta_0 + \\beta_1 d}
```

# Example
```jldoctest
julia> takata(20.0, 18.0, [20000.0, 400.0])
0.2571428571428571
```
"""
takata(d, h, θ) = d^2 * h / (θ[1] + θ[2] * d)
