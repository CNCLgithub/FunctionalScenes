
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
