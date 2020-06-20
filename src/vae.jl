using Distances

struct VAE{Tp<:TuringMvNormal, Te<:ConditionalMeanVarMvNormal, Td<:ConditionalMeanVarMvNormal}
    prior::Tp
    encoder::Te
    decoder::Td
end

Flux.@functor ConditionalMeanVarMvNormal
Flux.@functor VAE
Flux.params(m::ConditionalMeanVarMvNormal) = Flux.params(m.mapping)
Flux.params(m::VAE) = Flux.params(m.encoder, m.decoder)

function _kld_gaussian(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end

function (m::Distances.KLDivergence)(p::TuringMvNormal, q::TuringMvNormal)
    μ1, σ1 = mean(p), var(p)
    μ2, σ2 = mean(p), var(p)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

Distances.kl_divergence(p::TuringMvNormal, q::TuringMvNormal) = KLDivergence()(p, q)

function elbo(m::VAE, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(logpdf(m.decoder, x, z))
    kld = mean(kl_divergence(condition(m.encoder, x), m.prior))
    llh - β*kld
end
