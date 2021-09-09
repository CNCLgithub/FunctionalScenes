"""
Given a trace, returns the objective over paths
"""
function batch_og(tr::Gen.Trace)
    @>> get_retval(tr) begin
        first # get the tracker state matrix
        x -> occupancy_grid(x, sigma = 0.0, decay = 0.0)
    end
end
