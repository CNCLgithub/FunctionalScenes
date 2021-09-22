export sinkhorn_div

function sinkhorn_div(p::Matrix{Float64}, q::Matrix{Float64};
                      λ::Float64 = 1.0,
                      ε::Float64 = 0.01,
                      scale::Float64 = 1.0) where {K, V}
    # discrete measures
    np = size(p, 1)
    nq = size(q, 1)
    measure_p = fill(1.0/np, np)
    measure_q = fill(1.0/nq, nq)
    # TODO: col-wise more performant?
    dm = pairwise(Euclidean(), p, q, dims = 1)
    ot = sinkhorn_unbalanced(measure_p, measure_q, dm, λ, λ, ε)
    d = sum(ot .* dm)
    isnan(d) || d < 0. ? 0. : d
end
