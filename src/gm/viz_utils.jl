export viz_render, viz_gt, save_gt_image

using FileIO: save
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

function viz_grid(grid::Matrix{Float64}, title::String)
    grid = reverse(grid, dims = 1)
    UnicodePlots.brightcolors!()
    println(heatmap(grid, border = :none,
                    title = title,
                    colorbar_border = :none))
    return nothing
end

function save_gt_image(trace::Gen.Trace, path::String)
    state = get_retval(trace)
    params = first(get_args(trace))
    g = params.graphics
    translated = translate(params.gt, Int64[], cubes = true)
    batch = @pycall functional_scenes.render_scene_batch([translated], g)::PyObject
    batch = Array{Float64, 4}(batch.cpu().numpy())
    img = colorview(RGB, batch[1, :, :, :])
    save(path, img)
end
