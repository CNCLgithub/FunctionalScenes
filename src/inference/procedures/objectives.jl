export multires_path

"""
Given a trace, returns the objective over paths
"""
function multires_path(tr::Gen.Trace)
    # tracker state matrix
    trackers = @>> tr get_retval first
    model_params = get_args(tr)
    @unpack template = model_params
    @>> trackers begin
        a_star_paths(template) # path in vertex space
        transforms(trackers) # cartesian space (Nx2 Matrix{Float64})
    end
end
