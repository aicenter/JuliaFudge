using DrWatson
@quickactivate "papers_in_julia"

using Plots
pyplot()

include(srcdir("annotatedheatmap.jl"))

df = include(joinpath(@__DIR__, "collect.jl"))

best = sort!(df,"tst")[1,"path"]
@unpack model = load(best)

p1 = annotatedheatmap(model[1].Re, title="NPU (real)",
                      c=:balance, clim=(-1.5,1.5), aspect_ratio=:equal, colorbar=false)
p2 = annotatedheatmap(model[1].Im, title="NPU (imag)",
                      c=:balance, clim=(-1.5,1.5), aspect_ratio=:equal, colorbar=false)
p3 = annotatedheatmap(model[2].W, title="NAU",
                      c=:balance, clim=(-1.5,1.5), aspect_ratio=:equal, colorbar=false)
p = plot(p3,p2,p1,layout=grid(1,3,widths=[0.6,0.2,0.2]),size=(1000,400))

wsave(plotsdir("bestmodel.pdf"), p)

display(p)
