using Gen
using FunctionalScenes
using Images: colorview, RGB, channelview
using FileIO: save
# using ProfileView
using Profile
using BenchmarkTools
using StatProfilerHTML


function mytest()
    room_dims = (16, 16)
    entrance = [8,9]
    exits = [16*16 - 8]
    r = GridRoom(room_dims, room_dims, entrance, exits)
    r = add(r, Set(16 * 8 + 8))
    r = expand(r, 2)

    params = QuadTreeModel(;gt = r)

    # cm = choicemap()
    # cm[:trackers => (1, Val(:production)) => :produce] = true
    # for i = 1 : 4
    #     cm[:trackers => (i + 1, Val(:production)) => :produce] = i == 2
    # end

    (trace, ll) = generate(qt_model, (params,))
    # display(@benchmark generate($qt_model, ($params,), $cm) seconds=10 )
    # Profile.clear()
    # @profilehtml (trace, ll) = generate(qt_model, (params,), cm)
    # display(get_submap(get_choices(trace), :trackers))
    st = get_retval(trace)
    img = colorview(RGB, permutedims(st.img_mu, (3, 1, 2)))
    # img = channelview(st.img_mu)
    save("/spaths/tests/qt_gm.png", img)
    @show ll
    display(img)
    return nothing
end

mytest();
