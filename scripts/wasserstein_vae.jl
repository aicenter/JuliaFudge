using DrWatson
@quickactivate "JuliaFudge"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using PDMats
using Distributions
using ConditionalDists
using GenerativeModels
using MLDatasets

using ConditionalDists: SplitLayer

function train!(loss, ps, data, opt)
    training_loss = loss(first(data))
    iters = length(data)
    @withprogress name="Loss: $training_loss" for (i,x) in enumerate(data)
        gs = Flux.gradient(ps) do
          training_loss = loss(x)
          return training_loss
        end
        ProgressLogging.@logprogress "Loss: $training_loss" i/iters
        Flux.Optimise.update!(opt, ps, gs)
    end
end

function run(config)
    @unpack zlength, hdim = config
    hd2 = Int(hdim/2)

    train_x, _ = MNIST.traindata(Float32)
    flat_x = reshape(train_x, :, size(train_x,3))
    data = Flux.Data.DataLoader(flat_x, batchsize=200, shuffle=true)

    xlength = size(flat_x, 1)

    # conditional normal encoder
    enc_map  = Chain(Dense(xlength, hdim, relu),
                     Dense(hdim, hd2, relu),
                     SplitLayer(hd2, [zlength,zlength], [identity,softplus]))
    encoder = ConditionalMvNormal(enc_map)

    
    # conditional normal decoder
    dec_map  = Chain(Dense(zlength, hd2, relu),
                     Dense(hd2, hdim, relu),
                     SplitLayer(hdim, [xlength,1], σ))
    decoder = ConditionalMvNormal(dec_map)

    model = VAE(zlength, encoder, decoder)
    k = GenerativeModels.IMQKernel()
    loss(x) = mmd_rand(model,x,k)

    ps = Flux.params(model)
    opt = ADAM()
    
    for e in 1:30
        @info "Epoch $e" loss(flat_x)
        train!(loss, ps, data, opt)
    end
    return @dict(model)
end

res, _ = produce_or_load(datadir("mnist"),
                prefix="wasserstein-vae",
                Dict(:hdim=>512, :zlength=>2),
                run, force=false)
model = res[:model]

test_x, test_y = MNIST.testdata(Float32)


using Plots
#pyplot()
test_x = test_x[:,:,1:6]

plts = []
for i in 1:size(test_x,3)
    img = test_x[:,:,i]
    s1 = heatmap(img'[end:-1:1,:], title="truth")
    push!(plts, s1)
    x = reshape(img,:)
    rec(x) = mean(model.decoder, mean(model.encoder, x))
    x̂ = rec(x)
    rec = reshape(x̂, size(img))'[end:-1:1,:]
    s2 = heatmap(rec, title="rec")
    push!(plts, s2)
end
p1 = plot(plts...)
display(p1)

test_x, test_y = MNIST.testdata(Float32)
z = mean(model.encoder, reshape(test_x,:,size(test_x,3)))
p2 = scatter(z[1,:], z[2,:], c=test_y)
