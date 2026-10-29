# Stem taper equations: the diameter `d` at the height `h` of a tree of diameter at breast height
# `D` and total height `H`, with `z = h/H` the relative height. They are fitted by `fitnls` with the
# predictors `(:h, :D, :H)`, in this order, and applied by `taperdiameter`, `taperheight`,
# `taperedvolume` and `logassortment`.

"""
    kozak1969(h, D, H, θ)

Quadratic taper of Kozak, Munro & Smith (1969), `θ = (β₀, β₁, β₂)`:

```math
\\left(\\frac{d}{D}\\right)^2 = \\beta_0 + \\beta_1 z + \\beta_2 z^2, \\qquad z = \\frac{h}{H}
```

The quadratic turns negative near the top of some stems, where the diameter is taken as zero; with
those zeros the variance cannot grow with the mean, so it is fitted with `variance=:constant`.

# Example
```jldoctest
julia> kozak1969(10.0, 25.0, 28.0, [1.06, -1.82, 0.749])
17.775258688091476
```
"""
function kozak1969(h, D, H, θ)
  z = h / H
  y = θ[1] + θ[2] * z + θ[3] * z^2
  y > 0 ? D * sqrt(y) : zero(D * y)
end

"""
    schoepfer1966(h, D, H, θ)

Fifth-degree polynomial of Schöepfer (1966), `θ = (β₀, …, β₅)`:

```math
\\frac{d}{D} = \\beta_0 + \\beta_1 z + \\beta_2 z^2 + \\beta_3 z^3 + \\beta_4 z^4 + \\beta_5 z^5, \\qquad z = \\frac{h}{H}
```

# Example
```jldoctest
julia> schoepfer1966(10.0, 25.0, 28.0, [1.12, -3.07, 10.6, -19.7, 16.0, -4.94])
17.74510713541977
```
"""
schoepfer1966(h, D, H, θ) = D * evalpoly(h / H, θ)

"""
    ormerod1973(h, D, H, θ; href)

Taper of Ormerod (1973), a power of the length above the stem to the top, `θ = (β,)`:

```math
d = D \\left(\\frac{H - h}{H - h_{ref}}\\right)^{\\beta}
```

so that ``d = D`` at the height of reference `href` (given in the unit of the heights).

# Example
```jldoctest
julia> ormerod1973(10.0, 25.0, 28.0, [0.804]; href=1.3)
18.208075630475562
```
"""
ormerod1973(h, D, H, θ; href) = D * ((H - h) / (H - href))^θ[1]

"""
    demaerschalk1972(h, D, H, θ)

Taper of Demaerschalk (1972), compatible with the Schumacher–Hall volume equation, `θ = (β₀, …, β₃)`:

```math
d = \\beta_0 \\, D^{\\beta_1} (H - h)^{\\beta_2} H^{\\beta_3}
```

# Example
```jldoctest
julia> demaerschalk1972(10.0, 25.0, 28.0, [0.885, 0.903, 0.804, -0.665])
18.036908485467347
```
"""
demaerschalk1972(h, D, H, θ) = θ[1] * D^θ[2] * (H - h)^θ[3] * H^θ[4]

"""
    maxburkhart1976(h, D, H, θ; a1, a2)

Segmented taper of Max & Burkhart (1976): three quadratics joined at the relative heights `a1`
(upper) and `a2` (lower), usually about 0.75 and 0.1, `θ = (β₁, …, β₄)`:

```math
\\left(\\frac{d}{D}\\right)^2 = \\beta_1 (z - 1) + \\beta_2 (z^2 - 1) + \\beta_3 (a_1 - z)^2 I_1 + \\beta_4 (a_2 - z)^2 I_2,
\\qquad I_i = \\begin{cases} 1 & z \\le a_i \\\\ 0 & z > a_i \\end{cases}
```

Where the sum turns negative the diameter is taken as zero.

# Example
```jldoctest
julia> maxburkhart1976(10.0, 25.0, 28.0, [-3.7, 1.83, -1.81, 46.8]; a1=0.75, a2=0.1)
17.72428253347736
```
"""
function maxburkhart1976(h, D, H, θ; a1, a2)
  z = h / H
  y = θ[1] * (z - 1) + θ[2] * (z^2 - 1) + θ[3] * (a1 - z)^2 * (z <= a1) + θ[4] * (a2 - z)^2 * (z <= a2)
  y > 0 ? D * sqrt(y) : zero(D * y)
end

"""
    kozak1988(h, D, H, θ; p)

Variable-exponent taper of Kozak (1988), with the inflection point at the relative height `p`
(about 0.2 to 0.3), `θ = (a₀, a₁, a₂, b₁, …, b₅)`:

```math
d = a_0 D^{a_1} a_2^{D} X^{\\,b_1 z^2 + b_2 \\ln(z + 0.001) + b_3 \\sqrt{z} + b_4 e^{z} + b_5 D/H},
\\qquad X = \\frac{1 - \\sqrt{z}}{1 - \\sqrt{p}}
```

# Example
```jldoctest
julia> kozak1988(10.0, 25.0, 28.0, [0.57, 1.14, 0.995, 0.324, -0.0501, -0.251, 0.215, 0.151]; p=0.25)
18.14675554808669
```
"""
function kozak1988(h, D, H, θ; p)
  z = h / H
  X = (1 - sqrt(z)) / (1 - sqrt(p))
  θ[1] * D^θ[2] * θ[3]^D * X^(θ[4] * z^2 + θ[5] * log(z + 0.001) + θ[6] * sqrt(z) + θ[7] * exp(z) + θ[8] * D / H)
end

"""
    kozak2004(h, D, H, θ; href)

Variable-exponent taper of Kozak (2004), `θ = (a₀, a₁, a₂, b₁, …, b₆)`:

```math
d = a_0 D^{a_1} H^{a_2} X^{\\,b_1 z^4 + b_2 e^{-D/H} + b_3 X^{0.1} + b_4 / D + b_5 H^{Q} + b_6 X},
\\qquad X = \\frac{1 - z^{1/3}}{1 - (h_{ref}/H)^{1/3}}, \\qquad Q = 1 - z^{1/3}
```

with `href` the height of reference of `D` (in the unit of the heights).

# Example
```jldoctest
julia> kozak2004(10.0, 25.0, 28.0, [0.971, 1.03, -0.0248, 0.503, -0.744, 0.645, 3.44, 0.0159, -0.149]; href=1.3)
17.780063015820122
```
"""
function kozak2004(h, D, H, θ; href)
  z = h / H
  X = (1 - z^(1 / 3)) / (1 - (href / H)^(1 / 3))
  Q = 1 - z^(1 / 3)
  θ[1] * D^θ[2] * H^θ[3] * X^(θ[4] * z^4 + θ[5] / exp(D / H) + θ[6] * X^0.1 + θ[7] / D + θ[8] * H^Q + θ[9] * X)
end

"""
    bi2000(h, D, H, θ; href)

Trigonometric variable-form taper of Bi (2000), `θ = (β₀, …, β₆)`:

```math
d = D \\left(\\frac{\\ln \\sin(\\frac{\\pi}{2} z)}{\\ln \\sin(\\frac{\\pi}{2} \\frac{h_{ref}}{H})}\\right)^{
\\beta_0 + \\beta_1 \\sin(\\frac{\\pi}{2} z) + \\beta_2 \\cos(\\frac{3\\pi}{2} z) + \\beta_3 \\sin(\\frac{\\pi}{2} z)/z + \\beta_4 D + \\beta_5 z \\sqrt{D} + \\beta_6 z \\sqrt{H}}
```

with `href` the height of reference of `D` (in the unit of the heights). It is not defined at the
ground (`h = 0`).

# Example
```jldoctest
julia> bi2000(10.0, 25.0, 28.0, [1.15, 0.0788, 0.0249, -0.619, 0.00107, 0.0181, -0.0604]; href=1.3)
18.490060611970986
```
"""
function bi2000(h, D, H, θ; href)
  z = h / H
  s = sin(π / 2 * z)
  D * (log(s) / log(sin(π / 2 * href / H)))^(θ[1] + θ[2] * s + θ[3] * cos(3π / 2 * z) + θ[4] * s / z +
                                            θ[5] * D + θ[6] * z * sqrt(D) + θ[7] * z * sqrt(H))
end
