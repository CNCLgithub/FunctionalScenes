export mybroadcast

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

# struct Mybroadcast <: Gen.Distribution{Array{Float32, 3}} end

# const mybroadcast = Mybroadcast()

# function Gen.random(::Mybroadcast, mean::Array{Float32, 3}, noise::Array{Float32, 3})
#     broadcasted_normal(mean, noise)
#     # img = mean .+ randn(size(mean)) .* noise
#     # return img
# end

# function Gen.logpdf(::Mybroadcast, image::Array{Float32, 3},
#                     mean::Array{Float32, 3}, noise::Array{Float32, 3})
#     ll = Gen.logpdf(broadcasted_normal(image, mean, noise))
#     display(ll)
#     return ll
#     # var = noise * noise
#     # vec = (image-mean)[:]
#     # lpdf = -(vec' * vec)/(2.0 * var) - 0.5 * log(2.0 * pi * var)

#     # return lpdf
# end


# (::Mybroadcast)(mean, noise) = Gen.random(Mybroadcast(), mean, noise)

# Gen.has_output_grad(::Mybroadcast) = false
# Gen.logpdf_grad(::Mybroadcast, value::Set, args...) = (nothing,)
