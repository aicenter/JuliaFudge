using DrWatson
@quickactivate "JuliaFudge"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using LinearAlgebra
using Statistics
using NeuralArithmetic

include(srcdir("dataset.jl"))

niters = 20000
βl1    = 0
lr     = 0.005
hdim   = 6

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
