module FunctionalScenes

#################################################################################
# Dependencies
#################################################################################

using Gen
using JSON
using Graphs
using Setfield
using Statistics
using Parameters
using Gen_Compose
using Lazy: @>, @>>
using LinearAlgebra
using OptimalTransport
using OrderedCollections
using SimpleWeightedGraphs
using Base.Iterators: take
using FunctionalCollections

#################################################################################
# Runtime configuration
#################################################################################

using PyCall
const torch = PyNULL()
const mi = PyNULL()
const dr = PyNULL()
const fs_py = PyNULL()
const numpy = PyNULL()
function __init__()
    copy!(numpy, pyimport("numpy"))
    copy!(torch, pyimport("torch"))
    copy!(dr, pyimport("drjit"))
    copy!(mi, pyimport("mitsuba"))
    copy!(fs_py, pyimport("functional_scenes"))

    # mitsuba variant
    variants = @pycall mi.variants()::PyObject
    variant = "cuda_ad_rgb" in variants ? "cuda_ad_rgb" : "scalar_rgb"
    mi.set_variant(variant)
end

#################################################################################
# Module imports
#################################################################################

include("utils.jl")
include("dists.jl")
include("rooms/rooms.jl")
include("dgp/dgp.jl")
include("gm/gm.jl")
include("blender/blender.jl")
include("inference/inference.jl")

#################################################################################
# Load Gen functions
#################################################################################

@load_generated_functions

end # module
