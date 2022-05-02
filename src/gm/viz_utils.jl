export viz_render, viz_gt

using Images: colorview, RGB
using ImageInTerminal

function viz_render(trace::Gen.Trace)
    state = get_retval(trace)
    params = first(get_args(trace))
    g = params.graphics
    translated = translate(first(state.instances), Int64[], cubes = true)
    batch = @pycall functional_scenes.render_scene_batch([translated], g)::PyObject
    batch = Array{Float64, 4}(batch.cpu().numpy())
    display(colorview(RGB, batch[1, :, :, :]))
end

function viz_gt(trace::Gen.Trace)
    params = first(get_args(trace))
    g = params.graphics
    translated = translate(params.gt, Int64[], cubes = true)
    batch = @pycall functional_scenes.render_scene_batch([translated], g)::PyObject
    batch = Array{Float64, 4}(batch.cpu().numpy())
    display(colorview(RGB, batch[1, :, :, :]))
end
