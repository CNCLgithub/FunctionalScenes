using JSON

function _load_device()
    if torch.cuda.is_available()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else
        device = torch.device("cpu")
    end
    return device
end


function softmax(x)
    x = x .- maximum(x)
    exs = exp.(x)
    sxs = sum(exs)
    n = length(x)
    isnan(sxs) || iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
end

"""
    read_json(path)
    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    open(path, "r") do f
        global data
        data = JSON.parse(f)
    end

    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end
