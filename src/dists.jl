export broadcasted_bernoulli, broadcasted_uniform

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

function Gen.logpdf(::BroadcastedUniform, xs::AbstractArray{Float64},
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

Gen.has_output_grad(::BroadcastedUniform) = false
Gen.logpdf_grad(::BroadcastedUniform, value::Set, args...) = (nothing,)
