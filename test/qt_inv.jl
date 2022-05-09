using Gen
using JSON
using FunctionalScenes


function mytest()
    room_dims = (16, 16)
    entrance = [8,9]
    exits = [16*16 - 8]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    r = expand(r, 2)

    params = QuadTreeModel(;gt = r)

    cm = choicemap()
    cm[:trackers => (1, Val(:production)) => :produce] = true
    cm[:trackers => (3, Val(:aggregation)) => :mu] = 0.1
    for i = 1:4
        cm[:trackers => (i+1, Val(:production)) => :produce] = i == 1
        cm[:trackers => (Gen.get_child(2, i, 4), Val(:production)) => :produce] = false
    end
    display(cm) 

    trace, ll = generate(qt_model, (params,), cm) 
    st = get_retval(trace)
    FunctionalScenes.viz_room(st.instances[1])
    c = FunctionalScenes.qt_path_cost(trace)
    cm2 = choicemap()
    cm2[:trackers => (3, Val(:aggregation)) => :mu] = 0.9
    new_trace,_ = update(trace, cm2)
    ds = FunctionalScenes.downstream_selection(new_trace, 3)
    new_trace,_ = regenerate(new_trace, ds)
    @show trace[:trackers => (3, Val(:aggregation)) => :mu]
    @show new_trace[:trackers => (3, Val(:aggregation)) => :mu]
    @show sum(sum(trace[:instances]))
    @show sum(sum(new_trace[:instances]))
   
    # new_trace, w, d = vertical_move(trace, 3)  
    st = get_retval(new_trace)
    FunctionalScenes.viz_room(st.instances[1])
    @show c
    @show FunctionalScenes.qt_path_cost(trace) - c
    return nothing
end

mytest();
