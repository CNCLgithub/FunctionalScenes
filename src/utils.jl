using JSON
using UnicodePlots
using FileIO: save
using Colors
using Images: colorview
using ImageInTerminal

export save_img_array,
    softmax,
    softmax!


# index arrays using sets. Order doesn't matter
function Base.to_index(i::Set{T}) where {T}
    Base.to_index(collect(T, i))
end


#################################################################################
# Visuals
#################################################################################

function display_mat(m::Matrix{Float64};
                     rotate::Bool = true,
                     c1=colorant"black",
                     c2=colorant"white")
    img = weighted_color_mean.(m, c2, c1)
    img = rotate ? rotr90(img, 3) : img
    display(img)
    return nothing
end

function display_img(m::Array{Float64, 3})
    img = colorview(RGB, permutedims(m, (3,1,2)))
    display(img)
    return nothing
end


#################################################################################
# IO
#################################################################################

function save_img_array(array::Array, path::String)
    _array = permutedims(array, (3,1,2))
    img = colorview(RGB, _array)
    save(path, img)
end


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
    local data
    open(path, "r") do f
        data = JSON.parse(f)
    end

    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end

function _init_graphics()
    variants = @pycall mi.variants()::PyObject
    if "cuda_ad_rgb" in variants
        @pycall mi.set_variant("cuda_ad_rgb")::PyObject
    else
        @pycall mi.set_variant("scalar_rgb")::PyObject
    end
    return nothing
end

#################################################################################
# Math
#################################################################################

# function softmax(x; t::Float64 = 1.0)
#     x = x .- maximum(x)
#     exs = exp.(x ./ t)
#     sxs = sum(exs)
#     n = length(x)
#     isnan(sxs) || iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
# end


function softmax(x::Array{Float64}; t::Float64 = 1.0)
    out = similar(x)
    softmax!(out, x; t = t)
    return out
end

function softmax!(out::Array{Float64}, x::Array{Float64}; t::Float64 = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

function uniform_weights(x)::Vector{Float64}
    n = length(x)
    fill(1.0 / n, n)
end
# TODO: documentation
# deals with empty case
function safe_uniform_weights(x)::Vector{Float64}
    n = length(x) + 1
    ws = fill(1.0 / n, n)
    return ws
end

#################################################################################
# Room coordinate manipulation
#################################################################################

const unit_ci = CartesianIndex(1,1)

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


function refine_space(state, params)
    lower_dims = size(state)
    upper_dims = level_dims(params, next_lvl)
    kernel_dims = Int64.(upper_dims ./ lower_dims)

    lower_ref = CartesianIndices(lower_dims) # dimensions of coarse state
    kernel_ref = CartesianIndices(kernel_dims) # dimensions of coarse state

    upper_lref = LinearIndices(upper_dims)
    lower_lref = LinearIndices(lower_dims)
    kernel_lref = LinearIndices(kernel_dims)

    kp = prod(kernel_dims) # number of elements in kernel

    # iterate over coarse state kernel
    next_state = zeros(T, upper_dims)
    for lower in lower_ref
        # map together kernel steps
        i = lower_lref[lower]
        c = CartesianIndex((Tuple(lower) .- (1, 1)) .* kernel_dims)
        # @show c
        # iterate over scalars for each kernel sweep
        _sum = 0.
        for inner in kernel_ref
            # @show inner
            j = kernel_lref[inner] # index of inner
            idx = c + inner # cart index in refined space
            if j < kp # still retreiving from prop
                val = @read(u[:outer => i => :inner => j => :x],
                            :continuous)
                _sum += val
            else # solving for the final value
                val = state[lower] * kp - _sum
            end
            next_state[idx] = val
        end
    end
end
