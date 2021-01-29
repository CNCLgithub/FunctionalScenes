using FunctionalScenes
using FunctionalScenes: expand, cubify, translate
# using ProfileView


function mytest()
    room_dims = (11,20)
    entrance = [6]
    exits = [213, 217]
    r = Room(room_dims, room_dims, entrance, exits)
    r = expand(r, 2)
    # d = translate(r, false; cubes =true)
    # display(d)
    params = ModelParams(;
                         img_size = (240, 360),
                         instances = 10,
                         template = r,
                         feature_weights = "/datasets/alexnet_places365.pth.tar"
                         )
    println(params.device)
    model(params)
    # @profview model(params)
    @time model(params)
    # @profview model(params)
    return nothing
end

mytest();
