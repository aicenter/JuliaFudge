using DrWatson
@quickactivate "JuliaFudge"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using Distributions
using ConditionalDists
using MLDatasets

include(srcdir("splitlayer.jl"))
include(srcdir("vae.jl"))

function train!(loss, ps, data, opt)
    @progress for x in data
        gs = Flux.gradient(ps) do
          training_loss = loss(x)
          return training_loss
        end
        @info loss(first(data))
        Flux.Optimise.update!(opt, ps, gs)
    end
end

function run(config)
    @unpack zlength, hdim = config
    hd2 = Int(hdim/2)

    train_x, _ = MNIST.traindata(Float32)
    flat_x = reshape(train_x, :, size(train_x,3))
    data = Flux.Data.DataLoader(flat_x, batchsize=128, shuffle=true)

    xlength = size(flat_x, 1)
    
    # standard normal prior
    prior = MvNormal(zeros(Float32, zlength), ones(Float32, zlength))
    
    # conditional normal encoder
    enc_map  = Chain(Dense(xlength, hdim, relu),
                     Dense(hdim, hd2, relu),
                     SplitLayer(hd2, [zlength,zlength]))
    encoder = ConditionalMvNormal(enc_map)

    
    # conditional normal decoder
    dec_map  = Chain(Dense(zlength, hd2, relu),
                     Dense(hd2, hdim, relu),
                     SplitLayer(hdim, [xlength,xlength], σ))
    decoder = ConditionalMvNormal(dec_map)

    model = VAE(prior, encoder, decoder)
    # loss(x) = -elbo(model,x)
    function loss(x)
        z = mean(model.encoder, x)
        llh = mean(logpdf(model.decoder, x, z))

        -llh
    end
    # display(loss(first(data)))
    # error()

    ps = Flux.params(model)
    opt = ADAM(0.001)
    
    for e in 1:1
        @info "Epoch $e" loss(flat_x)
        train!(loss, ps, data, opt)
    end
    return @dict(model)
end

res, _ = produce_or_load(datadir("mnist"),
                Dict(:hdim=>512, :zlength=>2),
                run, force=true)
model = res[:model]

test_x, test_y = MNIST.testdata(Float32)

using Plots
pyplot()
test_x = test_x[:,:,1:6]

plts = []
for i in 1:size(test_x,3)
    img = test_x[:,:,i]
    p1 = heatmap(img'[end:-1:1,:], title="truth")
    push!(plts, p1)
    x = reshape(img,:)
    x̂ = model(x)
    rec = reshape(x̂, size(img))'[end:-1:1,:]
    p2 = heatmap(rec, title="rec")
    push!(plts, p2)
end
p1 = plot(plts...)

test_x, test_y = MNIST.testdata(Float32)
z = mean(model.encoder, reshape(test_x,:,size(test_x,3)))
p2 = scatter(z[1,:], z[2,:], c=test_y)
