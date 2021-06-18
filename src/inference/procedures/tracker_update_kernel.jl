@gen function random_walk_proposal(trace, tracker)
    addr = :trackers => tracker => :state
    {addr} ~ broadcasted_normal(trace[addr], 0.05)
end


@kern function tracker_kernel(trace, tracker)

    # update the level of a given tracker
    trace ~ mh(trace, split_merge_proposal, (tracker,), split_merge_involution)
    # random walk over state for that tracker
    trace ~ mh(trace, random_walk_proposal, (tracker,))

    # update room instances; currently using ancestral distribution
    selected = select_tracker_from_state(trace, tracker)
    trace ~ mh(trace, selected)
end
