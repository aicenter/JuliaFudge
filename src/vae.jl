using Distances
using Distributions
# using DistributionsAD
# 
# const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
#                          DistributionsAD.TuringDiagMvNormal,
#                          DistributionsAD.TuringScalMvNormal}

struct VAE{Tp<:MvNormal, Te<:ConditionalMvNormal, Td<:ConditionalMvNormal}
    prior::Tp
    encoder::Te
    decoder::Td
end

(m::VAE)(x) = mean(m.decoder, mean(m.encoder, x))

Flux.@functor ConditionalMvNormal
Flux.@functor VAE
Flux.params(m::ConditionalMvNormal) = Flux.params(m.mapping)
Flux.params(m::VAE) = Flux.params(m.encoder, m.decoder)

function _kld_gaussian(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end

function (m::Distances.KLDivergence)(p, q)
    μ1, σ1 = mean(p), var(p)
    μ2, σ2 = mean(p), var(p)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

Distances.kl_divergence(p, q) = KLDivergence()(p, q)

function elbo(m::VAE, x::AbstractArray; β=1)
    # sample latent codes
    z = rand(m.encoder, x)

    # reconstruction error
    llh = mean(logpdf(m.decoder, x, z))

    # KLD with `condition`ed encoder
    kld = mean(kl_divergence(condition(m.encoder, x), m.prior))

    llh - β*kld
end
