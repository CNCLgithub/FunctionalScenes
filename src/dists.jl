export broadcasted_bernoulli, broadcasted_uniform, labelled_categorical

@dist function labelled_categorical(xs)
    n = length(xs)
    probs = fill(1.0 / n, n)
    index = categorical(probs)
    xs[index]
end

@dist function id(x)
    probs = ones(1)
    xs = fill(x, 1)
    index = categorical(probs)
    xs[index]
end

struct BroadcastedBernoulli <: Gen.Distribution{AbstractArray{Bool}} end

const broadcasted_bernoulli = BroadcastedBernoulli()

function Gen.random(::BroadcastedBernoulli, ws::Array{Float64})
    bernoulli.(ws)
end

function Gen.logpdf(::BroadcastedBernoulli, xs::AbstractArray{Bool},
                    ws::AbstractArray{Float64})
    s = size(ws)
    Gen.assert_has_shape(xs, s;
                     msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    ll = 0
    for i in LinearIndices(s)
        ll += Gen.logpdf(bernoulli, xs[i], ws[i])
    end
    return ll
end

(::BroadcastedBernoulli)(ws) = Gen.random(BroadcastedBernoulli(), ws)

is_discrete(::BroadcastedBernoulli) = true
Gen.has_output_grad(::BroadcastedBernoulli) = false
Gen.logpdf_grad(::BroadcastedBernoulli, value::Set, args...) = (nothing,)




struct BroadcastedUniform <: Gen.Distribution{AbstractArray{Float64}} end

const broadcasted_uniform = BroadcastedUniform()

function Gen.random(::BroadcastedUniform, ws::AbstractArray{Tuple{Float64, Float64}})
    result = Array{Float64}(undef, size(ws)...)
    for i in LinearIndices(ws)
        a,b = ws[i]
        result[i] = uniform(a, b)
    end
    return result
end



function Gen.logpdf(::BroadcastedUniform,
                    xs::Float64,
                    ws::AbstractArray{Tuple{Float64, Float64}})
    Gen.logpdf(uniform, xs, ws[1]...)
end

function Gen.logpdf(::BroadcastedUniform,
                    xs::AbstractArray{Float64},
                    ws::AbstractArray{Tuple{Float64, Float64}})
    s = size(ws)
    Gen.assert_has_shape(xs, s;
                     msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    ll = 0
    for i in LinearIndices(s)
        ll += Gen.logpdf(uniform, xs[i], ws[i]...)
    end
    return ll
end

(::BroadcastedUniform)(ws) = Gen.random(BroadcastedUniform(), ws)

function logpdf_grad(::BroadcastedUniform,
                     xs::Union{AbstractArray{Float64}, Float64},
                     ws::AbstractArray{Tuple{Float64, Float64}})
    # s = size(ws)
    # Gen.assert_has_shape(xs, s;
    #                  msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    inv_diff = 0.
    for i in LinearIndices(s)
        low, high = ws[i]
        inv_diff += 1.0 / (high - low)
    end
    (0., inv_diff, -inv_diff)
end


is_discrete(::BroadcastedUniform) = false
has_output_grad(::BroadcastedUniform) = true
has_argument_grads(::BroadcastedUniform) = (true, true)



struct BroadcastedPiecewiseUniform <: Gen.Distribution{AbstractArray{Float64}} end


const broadcasted_piecewise_uniform = BroadcastedPiecewiseUniform()

function Gen.random(::BroadcastedPiecewiseUniform,
                    ws::AbstractArray{Tuple{Vector{Float64}, Vector{Float64}}, N}) where {N}
    result = Array{Float64}(undef, size(ws)...)
    for i in LinearIndices(ws)
        a,b = ws[i]
        result[i] = piecewise_uniform(a, b)
    end
    return result
end

function Gen.logpdf(::BroadcastedPiecewiseUniform,
                    xs::Float64,
                    ws::AbstractArray{Tuple})
    Gen.logpdf(piecewise_uniform, xs, ws[1]...)
end

function Gen.logpdf(::BroadcastedPiecewiseUniform,
                    xs::AbstractArray{Float64},
                    ws::AbstractArray{Tuple{Vector{Float64}, Vector{Float64}}, N}) where {N}
    s = size(ws)
    Gen.assert_has_shape(xs, s;
                     msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    ll = 0
    for i in LinearIndices(s)
        ll += Gen.logpdf(piecewise_uniform, xs[i], ws[i]...)
    end
    return ll
end

(::BroadcastedPiecewiseUniform)(ws) = Gen.random(BroadcastedPiecewiseUniform(), ws)

function logpdf_grad(::BroadcastedPiecewiseUniform,
                     xs::Union{AbstractArray{Float64}, Float64},
                     ws::AbstractArray{Tuple{Vector{Float64}, Vector{Float64}}, N}) where {N}
    # s = size(ws)
    # Gen.assert_has_shape(xs, s;
    #                  msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    a, b = 0., 0.
    for i in LinearIndices(s)
        _, _a, _b = Gen.logpdf_grad(piecewise_uniform, xs[i], ws[i]...)
        a += _a
        b += _b
    end
    (0., a, b)
end


is_discrete(::BroadcastedPiecewiseUniform) = false
has_output_grad(::BroadcastedPiecewiseUniform) = true
has_argument_grads(::BroadcastedPiecewiseUniform) = (true, true)
