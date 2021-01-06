module FunctionalScenes

using Gen
using Lazy
using GenRFS
using Statistics
using LightGraphs
using MetaGraphs
using LinearAlgebra

function __init__()
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
