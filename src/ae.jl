struct AutoEncoder{E,D}
    enc::E
    dec::D
end
Flux.@functor AutoEncoder

(m::AutoEncoder)(x) = m.dec(m.enc(x))
