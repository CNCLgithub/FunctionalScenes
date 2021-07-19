using JSON

#################################################################################
# IO
#################################################################################

function _load_device()
    if torch.cuda.is_available()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else
        device = torch.device("cpu")
    end
    return device
end

"""
    read_json(path)
    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    open(path, "r") do f
        global data
        data = JSON.parse(f)
    end

    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end


#################################################################################
# Math
#################################################################################

function softmax(x)
    x = x .- maximum(x)
    exs = exp.(x)
    sxs = sum(exs)
    n = length(x)
    isnan(sxs) || iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
end

#################################################################################
# Room coordinate manipulation
#################################################################################

"""
Is coordinate `a` adjacent to `b`?
"""
function is_next_to(a::CartesianIndex{2}, b::CartesianIndex{2})
    d = abs.(Tuple(a - b))
    # is either left,right,above,below
    d == (1, 0) || d == (0, 1)
end

"""
Takes a tile in an `mxn` space and \"expands\" by `factor`
"""
function up_scale_inds(src::CartesianIndices{2}, dest::CartesianIndices{2},
                       factor::Int64, vs::Vector{Int64})
    result = Array{Int64, 3}(undef, factor, factor, length(vs))
    for i = 1:length(vs)
        result[:, :, i] = up_scale_inds(src, dest, factor, vs[i])
    end
    vec(result)
end
function up_scale_inds(src::CartesianIndices{2}, dest::CartesianIndices{2},
                       factor::Int64, v::Int64)
    kern = CartesianIndices((1:factor, 1:factor))
    offset = CartesianIndex(1,1)
    dest_l = LinearIndices(dest)
    dest_l[(src[v] - offset) * factor .+ kern]
end
