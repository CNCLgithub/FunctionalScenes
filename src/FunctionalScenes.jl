module FunctionalScenes

using Gen
using Lazy
using GenRFS
using PyCall
using Statistics
using LightGraphs
using MetaGraphs
using LinearAlgebra
using Luxor

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
include("gen.jl")
include("blender/blender.jl")
include("paths.jl")

end # module
