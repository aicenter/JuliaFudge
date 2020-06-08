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

function result_dict(model::Chain, config::Dict)
    res = Dict{Symbol,Any}([(k,v) for (k,v) in config])
    res[:model] = model

    # training error
    x = 0.1f0:0.1f0:2
    y = 0.1f0:0.1f0:2
    xy = reduce(hcat, map(t->[t...], Iterators.product(x,y)))
    t = model(xy)
    res[:add_trn]  = mean(abs, t[1,:] - vec(f1(xy)))
    res[:mult_trn] = mean(abs, t[2,:] - vec(f2(xy)))
    res[:div_trn]  = mean(abs, t[3,:] - vec(f3(xy)))
    res[:sqrt_trn] = mean(abs, t[4,:] - vec(f4(xy)))

    # testing error
    x = -4.1f0:0.2f0:4f0
    y = -4.1f0:0.2f0:4f0
    xy = reduce(hcat, map(t->[t...], Iterators.product(x,y)))
    t = model(xy)
    res[:add_tst]  = mean(abs, t[1,:] - vec(f1(xy)))
    res[:mult_tst] = mean(abs, t[2,:] - vec(f2(xy)))
    res[:div_tst]  = mean(abs, t[3,:] - vec(f3(xy)))

    x = 0.1f0:0.1f0:4
    y = 0.1f0:0.1f0:4
    xy = reduce(hcat, map(t->[t...], Iterators.product(x,y)))
    t = model(xy)
    res[:sqrt_tst] = mean(abs, t[4,:] - vec(f4(xy)))

    return res
end

function run_npu(d::Dict)
    @unpack niters, βl1, lr, hdim = d

    model = Chain(NPU(2,hdim), NAU(hdim,4))
    ps = params(model)
    opt = ADAM(lr)
    loss(x,y) = Flux.mse(model(x),y) + βl1*norm(ps, 1)
    
    data = (generate() for _ in 1:niters)
    (x,y) = generate()
    
    cb = Flux.throttle(() -> (@info loss(x,y)), 1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,d)
end

@progress for run in 1:5
    res, _ = produce_or_load(datadir("arithmetic"),
                           Dict(:niters=>2000,
                                :βl1=>0,
                                :lr=>0.005,
                                :run=>run,
                                :hdim=>6),
                           run_npu,
                           prefix="npu",
                           force=false, digits=6)
end
