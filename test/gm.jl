using Gen
using FunctionalScenes
using FunctionalScenes: expand, cubify, translate
# using ProfileView


function mytest()
    room_dims = (6,6)
    entrance = [3]
    exits = [33]
    # room_dims = (11,20)
    # entrance = [6]
    # exits = [213, 217]
    r = Room(room_dims, room_dims, entrance, exits)
    # r = expand(r, 2)
    # d = translate(r, false; cubes =true)
    # display(d)
    params = ModelParams(;
                         thickness = (1,1),
                         dims = (2,2),
                         img_size = (240, 360),
                         instances = 10,
                         template = r,
                         feature_weights = "/datasets/alexnet_places365.pth.tar",
                         base_sigma = 10.0)
    println(params.device)

    # display(params.linear_ref)
    # display(FunctionalScenes.tracker_to_state(params, 3))
    s = FunctionalScenes.select_from_model(params, 3)
    # display(s)

    comp = complement(select(:viz))
    # # @profview model(params)
    trace, ll = generate(model, (params,))
    # choices = get_choices(trace)
    choices = get_selected(get_choices(trace), comp)
    open("/project/output/a.txt", "w") do io
        Base.show(io, "text/plain", choices)
    end
    # display(choices[:viz])
    # s = StaticSelection(select(:active => 1,
    #                            :trackers => 1 => :state))
    # display(s)
    nt, ll, diff = regenerate(trace, s)
    # choices = get_choices(nt)
    # display(choices[:viz])
    # display(diff)
    nchoices = get_selected(get_choices(nt), comp)
    open("/project/output/b.txt", "w") do io
        Base.show(io, "text/plain", nchoices)
    end
    println("ll $(ll)")
    llvz = project(nt, select(:viz))
    println("viz $(llvz)")
    println("s3 $(project(nt, s))")
    s2 = FunctionalScenes.select_from_model(params, 2)
    println("s2 $(project(nt, s2))")
    s1 = FunctionalScenes.select_from_model(params, 1)
    println("s1 $(project(nt, s1))")
    s4 = FunctionalScenes.select_from_model(params, 4)
    println("s4 $(project(nt, s4))")
    # @time model(params)
    # @profview model(params)
    # display(get_selected(get_choices(nt), s1))
    return nothing
end

mytest();
