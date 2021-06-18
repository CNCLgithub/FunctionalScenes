@kern function tracker_kernel(trace, tracker)

    # update the level of a given tracker
    trace ~ mh(trace, )
    for i in 1:trace[:n]
        trace ~ mh(trace, random_walk_proposal, (i,))
    end

    # repeatedly pick a random x and do a random walk update on it
    if trace[:n] > 0
        for rep in 1:10
            let i ~ uniform_discrete(1, trace[:n])
                trace ~ mh(trace, random_walk_proposal, (i,))
            end
        end
    end

    # remove the last x, or add a new one, a random number of times
    let n_add_remove_reps ~ uniform_discrete(0, max_n_add_remove)
        for rep in 1:n_add_remove_reps
            trace ~ mh(trace, add_remove_proposal, (), add_remove_involution)
        end
    end
end
