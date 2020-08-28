using Distances
using Distributions


struct VAE{P<:MvNormal,E<:ConditionalMvNormal,D<:ConditionalMvNormal}
    prior::P
    encoder::E
    decoder::D
end

Flux.@functor VAE

function Flux.trainable(m::VAE)
    (encoder=m.encoder, decoder=m.decoder)
end

function VAE(zlength::Int, enc::ConditionalMvNormal, dec::ConditionalMvNormal)
    W = first(Flux.params(enc))
    μ = fill!(similar(W, zlength), 0)
    σ = fill!(similar(W, zlength), 1)
    prior = MvNormal(μ, σ)
    VAE(prior, enc, dec)
end

function elbo(m::VAE, x::AbstractArray; β=1)
    # sample latent
    z = rand(m.encoder, x)

    # reconstruction error
    llh = mean(logpdf(m.decoder, x, z))

    # KLD with `condition`ed encoder
    kld = mean(kl_divergence(condition(m.encoder, x), m.prior))

    llh - β*kld
end

# mmd via IPMeasures
# """
#     mmd_mean(m::AbstractVAE, x::AbstractArray, k[; distance])
# 
# Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Uses mean of encoded data.
# """
# mmd_mean(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
#     mmd(k, mean(m.encoder, x), rand(m.prior, size(x, 2)), distance)
# 
# """
#     mmd_rand(m::AbstractVAE, x::AbstractArray, k[; distance])
# 
# Maximum mean discrepancy of a VAE given data `x` and kernel function `k(x,y)`. Samples from the encoder.
# """
# mmd_rand(m::AbstractVAE, x::AbstractArray, k; distance = IPMeasures.pairwisel2) = 
#     mmd(k, rand(m.encoder, x), rand(m.prior, size(x, 2)), distance)

function Base.show(io::IO, m::VAE)
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    e = repr(m.encoder)
    e = sizeof(e)>70 ? "($(e[1:70-3])...)" : e
    d = repr(m.decoder)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(nameof(typeof(m))):
     prior   = $(p)
     encoder = $(e)
     decoder = $(d)
    """
    print(io, msg)
end


# TODO: use IPMeasures instead
function _kld_gaussian(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end

function (m::Distances.KLDivergence)(p, q)
    μ1, σ1 = mean(p), var(p)
    μ2, σ2 = mean(q), var(q)
    _kld_gaussian(μ1, σ1, μ2, σ2)
end

Distances.kl_divergence(p, q) = KLDivergence()(p, q)
