export broadcasted_bernoulli

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

(::BroadcastedBernoulli)(ws) = Gen.random(BroadcastedDist(), ws)

Gen.has_output_grad(::BroadcastedBernoulli) = false
Gen.logpdf_grad(::BroadcastedBernoulli, value::Set, args...) = (nothing,)
