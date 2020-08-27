struct SplitLayer
    layers::Tuple
end

function SplitLayer(input::Int, outputs::Array{Int,1}, act=abs)
    SplitLayer(Tuple(Dense(input,out,act) for out in outputs))
end

function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end

Flux.@functor SplitLayer
