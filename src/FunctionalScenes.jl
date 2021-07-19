module FunctionalScenes

#################################################################################
# Dependencies
#################################################################################

using Gen
using JSON
using GenRFS
using PyCall
using Statistics
using MetaGraphs
using Gen_Compose
using LightGraphs
using LinearAlgebra
using OptimalTransport
using OrderedCollections
using Parameters: @with_kw
using Base.Iterators: take
using Lazy: @>, @>>, lazymap, flatten

#################################################################################
# Runtime configuration
#################################################################################

const torch = PyNULL()
const functional_scenes = PyNULL()
function __init__()
    copy!(torch, pyimport("torch"))
    copy!(functional_scenes, pyimport("functional_scenes"))
end

#################################################################################
# Module imports
#################################################################################

include("utils.jl")
include("dists.jl")
include("room.jl")
include("furniture.jl")
include("gm/gm.jl")
include("blender/blender.jl")
include("paths.jl")
include("inference/inference.jl")

#################################################################################
# Load Gen functions
#################################################################################

@load_generated_functions

end # module
