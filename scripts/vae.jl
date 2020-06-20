using DrWatson
@quickactivate "JuliaFudge"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using ConditionalDists
using MLDatasets

include(srcdir("vae.jl"))

function train!(loss, ps, data, opt)
    @withprogress for d in data
        gs = Flux.gradient(ps) do
          training_loss = loss(d...)
          return training_loss
        end
        @logprogress training_loss
        Flux.Optimise.update!(opt, ps, gs)
    end
end


function run(config)
    @unpack zlength, hdim = config

    train_x, train_y = MNIST.traindata()
    test_x,  test_y  = MNIST.testdata()
    flat_x = reshape(train_x, :, size(train_x,3))
    data = Flux.Data.DataLoader(flat_x, batchsize=128, shuffle=true)

    xlength = size(flat_x, 1)
    
    # standard normal prior
    prior = TuringMvNormal(zeros(Float32, zlength), ones(Float32, zlength))
    
    # conditional normal encoder
    enc_dist = TuringMvNormal(zeros(Float32, zlength), ones(Float32, zlength))
    enc_map  = Chain(Dense(xlength, hdim, relu),
                     Dense(hdim, zlength))
    encoder = ConditionalMeanVarMvNormal(enc_dist, enc_map)
    
    # conditional normal decoder
    dec_dist = TuringMvNormal(zeros(Float32, xlength), ones(Float32, xlength))
    dec_map  = Chain(Dense(zlength, hdim, relu),
                     Dense(hdim, xlength))
    decoder = ConditionalMeanVarMvNormal(dec_dist, dec_map)
    
    model = VAE(prior, encoder, decoder)
    ps = params(model)
    opt = ADAM()
    loss(x) = elbo(model,x)
    
    train!(loss, ps, data, opt)
    return @dict(model)
end

produce_or_load(datadir("mnist"),
                Dict(:hdim=>512, :zlength=>2),
                run)
