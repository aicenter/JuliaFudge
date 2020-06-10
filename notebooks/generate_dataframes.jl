# this is for generating the experimental data frames

using CSV # table loading/saving
using MLJ # Julia machine learning models
using Plots
using DataFrames

# load the needed models
@load KNNClassifier
@load XGBoostClassifier

"""
    collect_results(modelf, params, X, y)

Returns the cross entropy, Brier score and AUC for a given data, model and parameters.
"""
function evaluate_model(modelf, params, X, y)
    m = modelf(;params...) # create the model with params
    res = evaluate(m, X, y, resampling=CV(shuffle=true), measure=[cross_entropy, auc], verbosity=0) # get results - 3 scores
    return res.measurement # return results
end

"""
    test_params(modelf, param_list, X, y)

This function iterates a model over parameters and given data.
"""
function test_params(modelf, param_list, X, y, datasetname)
    # create an empty dataframe
    res = DataFrame(
        dataset = [],
        model = [],
        parameters = [],
        fold = [],
        cross_entropy = [],
        auc = []
        )
    # now compute the results for all parameter values
    for params in param_list
        # loop over different initializations
        for fold in 1:3
            # if the computation fails, return Missing values instead of crashing
            measure_vals = 
                try 
                    evaluate_model(modelf, params, X, y)
                catch e
                    (missing, missing)
                end
            modelstring = string(modelf) # create the model string
            paramstring = reduce((x,y)->"$x, $y", map(x->"$(x[1])=$(x[2])", params)) # create the string of parameters
            push!(res, (datasetname, modelstring, paramstring, fold, measure_vals...))
        end
    end
    return res
end

# create a folder where to save the data
savepath = "./data"
mkpath(savepath)

# run the KNN classifier on 2 datasets
modelf = KNNClassifier
param_list = map(x-> [:K=>x], [1, 11, 101, 301])

X, y = @load_iris
CSV.write(joinpath(savepath, "knn_iris.csv"), test_params(modelf, param_list, X, y, "iris"))
X, y = @load_crabs
CSV.write(joinpath(savepath, "knn_crabs.csv"), test_params(modelf, param_list, X, y, "crabs"))

# run the XGBoost classifier on the 2 datasets
modelf = XGBoostClassifier
param_list = map(x-> [:max_depth=>x], 2:2:8)

X, y = @load_iris
CSV.write(joinpath(savepath, "xgboost_iris.csv"), test_params(modelf, param_list, X, y, "iris"))
X, y = @load_crabs
CSV.write(joinpath(savepath, "xgboost_crabs.csv"), test_params(modelf, param_list, X, y, "crabs"))

# add some missing values to make the final analysis a bit more interesting
df = DataFrame(CSV.read(joinpath(savepath, "knn_crabs.csv")))
df[!, :auc][1] = missing
df[!, :auc][5] = missing
df[!, :cross_entropy][6] = missing
CSV.write(joinpath(savepath, "knn_crabs.csv"), df)