using DrWatson
@quickactivate "JuliaFudge"

# for loading...
using Flux
using NeuralArithmetic

using DataFrames

name(m::NPU) = "NPU"
name(m::RealNPU) = "RealNPU"
name(m::Dense) = "Dense"
name(m::NALU) = "NALU"
name(m::NMU) = "NMU"
name(m::iNALU) = "iNALU"
name(m::Chain) = name(m[1])

function delete_from_savename(path,key)
    (dir,dict,_) = parse_savename(path)
    delete!(dict, key)
    joinpath(dir, savename(dict,digits=20))
end

function collect_folder!(folder::String)
  sum_tst(data) = sum([data[:add_tst], data[:mult_tst],
                       data[:div_tst], data[:sqrt_tst]])
  _df = collect_results!(datadir(folder), black_list=[:model,:tst],
                         special_list=[:model=>data->name(data[:model]),
                                       :tst  =>data->sum_tst(data)])
  _df.hash = delete_from_savename.(_df.path, "run")
  return _df
end

df = collect_folder!(datadir("arithmetic"))
