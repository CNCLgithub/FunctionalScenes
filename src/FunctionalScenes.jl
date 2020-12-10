module FunctionalScenes

using Gen
using Lazy
using GenRFS
using LightGraphs
using MetaGraphs
using UnicodePlots

function __init__()
    # needed to deal with Gen static functions
    @load_generated_functions
end

include("room.jl")
include("furniture.jl")

end # module
