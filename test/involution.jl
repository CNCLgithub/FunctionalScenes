using Gen
using FunctionalScenes


function test()

    room_dims = (22,40)
    entrance = [11]
    exits = [869]
    r = Room(room_dims, room_dims, entrance, exits)
    weights = ones(steps(r))
    new_r = last(furniture_chain(10, r, weights))
    params = ModelParams(;
                         gt = new_r,
                         dims = (6, 6),
                         img_size = (240, 360),
                         instances = 10,
                         template = r,
                         feature_weights = "/datasets/alexnet_places365.pth.tar",
                         base_sigma = 10.0)

    @show params
    trace, ll = generate(model, (params,))

    println("ORIGINAL TRACE")
    display(trace[:trackers => 1 => :state])
    @show ll

    # mh(trace, split_merge_proposal, (1,), split_merge_involution)

    println("\n\n\nNEW TRACE")
    trace_translator = Gen.SymmetricTraceTranslator(split_merge_proposal, (1,), split_merge_involution)
    (new_trace, log_weight) = trace_translator(trace; check = false)

    display(new_trace[:trackers => 1 => :state])

    new_lvl, new_state = new_trace[:trackers => 1]
    @show log_weight



    cm = choicemap((:trackers => 1 => :level, new_lvl),
                   (:trackers => 1 => :state, new_state))
    trace, ll = generate(model, (params,), cm)
    @show ll
end

test();
