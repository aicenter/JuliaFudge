using Flux
using LinearAlgebra
using Statistics
using NeuralArithmetic

f1(x::Array) = reshape(x[1,:] .+ x[2,:], 1, :)
f2(x::Array) = reshape(x[1,:] .* x[2,:], 1, :)
f3(x::Array) = reshape(x[1,:] ./ x[2,:], 1, :)
f4(x::Array) = reshape(sqrt.(x[1,:]), 1, :)
f(x::Array)  = cat(f1(x),f2(x),f3(x),f4(x),dims=1)

function generate()
    x = rand(Float32, 2, 100) .* 1.9f0 .+ 0.1f0
    y = f(x)
    (x,y)
end

niters = 20000
βl1    = 0
lr     = 0.005
hdim   = 6

model = Chain(NPU(2,hdim), NAU(hdim,4))
ps = params(model)
opt = ADAM(lr)
loss(x,y) = Flux.mse(model(x),y) + βl1*norm(ps, 1)

data = (generate() for _ in 1:niters)
(x,y) = generate()

cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
Flux.train!(loss, ps, data, opt, cb=cb)
