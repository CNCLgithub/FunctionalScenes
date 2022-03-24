module FunctionalScenes

#################################################################################
# Dependencies
#################################################################################

using Gen
using JSON
using Graphs
using PyCall
using Statistics
using Parameters
# using Gen_Compose
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

# const torch = PyNULL()
# const functional_scenes = PyNULL()
# function __init__()
#     copy!(torch, pyimport("torch"))
#     copy!(functional_scenes, pyimport("functional_scenes"))
# end

#################################################################################
# Module imports
#################################################################################

include("utils.jl")
include("dists.jl")
include("rooms/rooms.jl")
include("dgp/dgp.jl")
# include("gm/gm.jl")
# include("blender/blender.jl")
# include("inference/inference.jl")

#################################################################################
# Load Gen functions
#################################################################################

@load_generated_functions

end # module
