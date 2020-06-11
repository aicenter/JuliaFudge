using DrWatson
@quickactivate "JuliaFudge"   # start with correct project environment
                              # I recommend just starting with `julia --project`
                              # but @quickactivate does not hurt

using LinearAlgebra
using Statistics
using Flux                    # ML Library
using NeuralArithmetic        # Custom layers

include(srcdir("dataset.jl")) # defines `generate` to produce our data
                              # julia>?generate to find out what it does!

# hyper parameters
niters = 20000
βl1    = 0
lr     = 0.005
hdim   = 6

"""
    run_npu(d::Dict)

Trains an NPU to learn the task defined by the `generate` function.
"""
function run_npu(d::Dict)
    @unpack niters, βl1, lr, hdim = d

    model = Chain(NPU(2,hdim), NAU(hdim,4))
    ps = params(model)
    opt = ADAM(lr)
    loss(x,y) = Flux.mse(model(x),y) + βl1*norm(ps, 1)
    
    data = (generate() for _ in 1:niters)
    (x,y) = generate()
    
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
end

run = 1
produce_or_load(datadir("arithmetic"),
                @dict(niters, βl1, lr, hdim, run),
                run_npu,
                prefix="npu",
                digits=6,
                force=false)
