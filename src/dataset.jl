f1(x::Array) = reshape(x[1,:] .+ x[2,:], 1, :)
f2(x::Array) = reshape(x[1,:] .* x[2,:], 1, :)
f3(x::Array) = reshape(x[1,:] ./ x[2,:], 1, :)
f4(x::Array) = reshape(sqrt.(x[1,:]), 1, :)
f(x::Array)  = cat(f1(x),f2(x),f3(x),f4(x),dims=1)

function generate()
    x = rand(Float32, 2, 100) .* 2 .+ 0.1f0
    y = f(x)
    (x,y)
end

addloss(model,x::Real,y::Real)  = abs(model([x,y])[1] - f1([x,y])[1])
multloss(model,x::Real,y::Real) = abs(model([x,y])[2] - f2([x,y])[1])
divloss(model,x::Real,y::Real)  = abs(model([x,y])[3] - f3([x,y])[1])
sqrtloss(model,x::Real,y::Real) = abs(model([x,y])[4] - f4([x,y])[1])
