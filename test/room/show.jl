using FunctionalScenes: Room, k_shortest_paths, furnish, steps, add, furniture_chain, entrance, exits
import FunctionalScenes: expand
using Gen
r = Room((10,20), (10,20), [5], [192]);
j = 3
display(exits(r))
ps = k_shortest_paths(r, j, 5, 192)
#categorical(1)
@show ps
for p in ps
    display((r, p))
end
probs = fill(1.0 / j, j)
index = categorical(probs)
#display(ps[index])
#furniture_chain = Gen.Unfold(furniture_step)
k = 12
factor = 1
weights = zeros(steps(r))
start_x = Int(last(steps(r)) * 0.4)
stop_x = last(steps(r)) - 2
start_y = 2
stop_y = first(steps(r)) - 1
weights[start_y:stop_y, start_x:stop_x] .= 1.0
weights[ps[index]] .= 0
new_r = last(furniture_chain(k, r, weights))
new_r = expand(new_r, factor)
dist = ps[j]
new_room = (new_r, dist)

#display(new_room);
