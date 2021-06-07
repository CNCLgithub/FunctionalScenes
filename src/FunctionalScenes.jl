module FunctionalScenes

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
using Lazy: @>, @>>, lazymap, flatten

const torch = PyNULL()
const functional_scenes = PyNULL()
function __init__()
    copy!(torch, pyimport("torch"))
    copy!(functional_scenes, pyimport("functional_scenes"))
    # needed to deal with Gen static functions
    @load_generated_functions
end

include("dists.jl")
include("room.jl")
include("utils.jl")
include("furniture.jl")
include("gm/gm.jl")
include("blender/blender.jl")
include("paths.jl")
include("inference/inference.jl")

end # module
