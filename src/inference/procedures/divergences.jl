export sinkhorn_div

using Distances

function cart_dm(p::Matrix{Float64})
    s1, s2 = size(p)
    ds = zeros(length(p), length(p))
    lis = LinearIndices(p)
    for i = 1:s1, j = 1:s2, a = 1:s1, b = 1:s2
        x = lis[i, j]
        y = lis[a, b]
        ds[x, y] = sqrt((i - a)^2 + (j - b)^2)
        # ds[y, x] = sqrt((i - a)^2 + (j - b)^2)
    end
    @SVector vec(ds)
end

const dm_32x32 = cart_dm(zeros(32, 32))

function sinkhorn_div(p::SMatrix{32, 32, Float64},
                      q::SMatrix{32, 32, Float64};
                      λ::Float64 = 1.0,
                      ε::Float64 = 0.01,
                      scale::Float64 = 1.0) where {K, V}
    # discrete measures
    measure_p = vec(p) ./ sum(p)
    measure_q = vec(q) ./ sum(q)
    ot = sinkhorn_unbalanced(measure_p, measure_q, dm_32x32, λ, λ, ε)
    d = sum(ot .* dm)
    isnan(d) || d < 0. ? 0. : d
end
function sinkhorn_div(p::Matrix{Float64}, q::Matrix{Float64};
                      λ::Float64 = 1.0,
                      ε::Float64 = 0.01,
                      scale::Float64 = 1.0) where {K, V}
    # discrete measures
    measure_p = vec(p) ./ sum(p)
    measure_q = vec(q) ./ sum(q)
    # TODO: col-wise more performant?
    dm = cart_dm(p)
    ot = sinkhorn_unbalanced(measure_p, measure_q, dm, λ, λ, ε)
    d = sum(ot .* dm)
    isnan(d) || d < 0. ? 0. : d
end
