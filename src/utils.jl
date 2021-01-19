function swap_tiles!(g::PathGraph, p::Tuple{Tile, Tile})
    x,y = p
    a = get_prop(g, y, :type)
    b = get_prop(g, x, :type)
    set_prop!(g, x, :type, a)
    set_prop!(g, y, :type, b)
    return nothing
end

function connected(g::PathGraph, v::Tile)::Set{Tile}
    s = @>> v bfs_tree(g) edges collect induced_subgraph(g) last Set
    isempty(s) ? Set([v]) : s
end

# TODO: implement me!
function bitmap_render(r::Room)::Matrix{Bool}
    # ratio = [33, 40] # rows by columns (y, x)
    image_size = (36,24) # vertical horizonal
    f = 50 # focal length
    pixels_size = (720,480)
    bm = fill(false, pixels_size)
    serialized = translate(r,false)
    camera = serialized[:camera]
    camera_pos = camera[:position] # xyz of camera
    camera_rot = camera[:orientation] # xyz rot in radians
    objects = serialized[:objects] # both walls and furniture
    s_x = sin(camera_rot[1])
    c_x = cos(camera_rot[1])
    furniture = filter(x -> x[:appearance] == :blue, objects)
    for fur in furniture
        pos = fur[:position] # x,y,z
        dims = fur[:dims]
        upper_left_1 = [pos[1]+dims[1]/2,pos[2]-dims[2]/2, pos[3]+dims[3]/2]
        upper_right_1 = [pos[1]-dims[1]/2,pos[2]-dims[2]/2, pos[3]+dims[3]/2]
        lower_left_1 = [pos[1]+dims[1]/2,pos[2]+dims[2]/2, pos[3]-dims[3]/2]
        lower_right_1 = [pos[1]-dims[1]/2,pos[2]+dims[2]/2, pos[3]-dims[3]/2]
        upper_left = [pos[1]-dims[1]/2,pos[2]+dims[2]/2, pos[3]+dims[3]/2]
        upper_right= [pos[1]+dims[1]/2,pos[2]+dims[2]/2, pos[3]+dims[3]/2]
        lower_left = [pos[1]-dims[1]/2,pos[2]-dims[2]/2, pos[3]-dims[3]/2]
        lower_right =  [pos[1]+dims[1]/2,pos[2]-dims[2]/2, pos[3]-dims[3]/2]
        
        # first, put furnitures position into camera coordinates
        upper_left_camera = coordinates_to_camera(upper_left,camera_pos,camera_rot)
        upper_right_camera = coordinates_to_camera(upper_right,camera_pos,camera_rot)
        lower_left_camera = coordinates_to_camera(lower_left,camera_pos,camera_rot)
        lower_right_camera = coordinates_to_camera(lower_right,camera_pos,camera_rot)
        upper_left_camera_1 = coordinates_to_camera(upper_left_1,camera_pos,camera_rot)
        upper_right_camera_1 = coordinates_to_camera(upper_right_1,camera_pos,camera_rot)
        lower_left_camera_1 = coordinates_to_camera(lower_left_1,camera_pos,camera_rot)
        lower_right_camera_1 = coordinates_to_camera(lower_right_1,camera_pos,camera_rot)
     
        # second, perspective project to pixel space
        upper_left_image = coordinates_to_pixels(f,upper_left_camera,image_size,pixels_size)
        upper_right_image = coordinates_to_pixels(f,upper_right_camera,image_size,pixels_size)
        lower_left_image = coordinates_to_pixels(f,lower_left_camera,image_size,pixels_size)
        lower_right_image = coordinates_to_pixels(f,lower_right_camera,image_size,pixels_size)
        upper_left_image_1 = coordinates_to_pixels(f,upper_left_camera_1,image_size,pixels_size)
        upper_right_image_1 = coordinates_to_pixels(f,upper_right_camera_1,image_size,pixels_size)
        lower_left_image_1 = coordinates_to_pixels(f,lower_left_camera_1,image_size,pixels_size)
        lower_right_image_1 = coordinates_to_pixels(f,lower_right_camera_1,image_size,pixels_size)
      
        f_bm = fill(false, pixels_size)
		vs = [upper_left_image, upper_right_image,lower_left_image,lower_right_image,upper_left_image_1, upper_right_image_1,lower_left_image_1,lower_right_image_1]
		vs_x = [upper_left_image[1], upper_right_image[1],lower_left_image[1],lower_right_image[1],upper_left_image_1[1], upper_right_image_1[1],lower_left_image_1[1],lower_right_image_1[1]]
		vs_z = [upper_left_image[2], upper_right_image[2],lower_left_image[2],lower_right_image[2],upper_left_image_1[2], upper_right_image_1[2],lower_left_image_1[2],lower_right_image_1[2]]
        polys = Point[]
	    notpolys = Point[]
		left_point = Point[]
		right_point = Point[]
		upper_point = Point[]
    	lower_point = Point[] 
		for ps in vs
			if ps[1] == minimum(vs_x)
				push!(left_point,ps)
				if length(left_point)>1
					left_upper = Point(minimum(vs_x),maximum(getindex.(left_point,2)))
					left_lower = Point(minimum(vs_x),minimum(getindex.(left_point,2)))
					left_point = Point[left_upper,left_lower]
				end
			end
			if  ps[2] == minimum(vs_z)
				push!(lower_point,ps)
				if length(lower_point)>1
					lower_right = Point(maximum(getindex.(upper_point,1)),minimum(vs_z))
					lower_left = Point(minimum(getindex.(upper_point,1)),minimum(vs_z))
					lower_point = Point[lower_left,lower_right]
				end
			end
			if  ps[1] == maximum(vs_x)
				push!(right_point,ps)
				if length(right_point)>1
					right_lower = Point(maximum(vs_x),minimum(getindex.(right_point,2)))
					right_upper = Point(maximum(vs_x),maximum(getindex.(right_point,2)))
					right_point = Point[right_lower,right_upper]
				end
			end		
			if  ps[2] == maximum(vs_z)
				push!(upper_point,ps)
				if length(upper_point)>1
					upper_right = Point(maximum(getindex.(upper_point,1)),maximum(vs_z))
					upper_left = Point(minimum(getindex.(upper_point,1)),maximum(vs_z))
					upper_point = Point[upper_right,upper_left]
				end
			end
		#	else
		#		push!(notpolys,ps)
		#	end
		end
		#polys = Point[left_point,lower_point,right_point,upper_point]
		for ps in left_point
			push!(polys,ps)
		end
		for ps in lower_point
			push!(polys,ps)
		end
		for ps in right_point
			push!(polys,ps)
		end
		for ps in upper_point
			push!(polys,ps)
		end
		#for ps in notpolys
		#	if !isinside(ps,polys)
		#		push!(polys,ps)
		#	end
		#end
	#end
#	println([min_x,max_x, round(Int,upper_left_image[2]),lower_left_image[2]])
#	for i in round(Int, min_x):round(Int,max_x)
#		for j in round(Int,upper_left_image[2]):round(Int,lower_left_image[2])
#   			point = Point(i,j)
#			if i < pixels_size[1] && j < pixels_size[2] && isinside(point,vs)
#      				f_bm[i,j] = true
		#	end
	  #	end
	#end
	polys = unique(polys)
	#println(polys)
	#println(vs)
	for i in 1:720
		for j in 1: 480
			point = Point(i,j)
			if isinside(point,polys)
				f_bm[i,j] = true
			end
		end
	end
    bm = bm .| f_bm
  end
return bm
end


function coordinates_to_camera(pos,camera_pos,camera_rot)::Array{Float64, 1}
    # first, put furnitures position into camera coordinates
    s_x = sin(camera_rot[1])
    c_x = cos(camera_rot[1])
    s_y = sin(camera_rot[2])
    c_y = cos(camera_rot[2])
    s_z = sin(camera_rot[3])
    c_z = cos(camera_rot[3])
    # x,y,z in camera coordinates after translation
    camera_x= pos[1] - camera_pos[1]
    camera_y= pos[2] - camera_pos[2]
    camera_z= pos[3] - camera_pos[3]
    # apply a rotation of camera with respect to x-axis
    d_x = c_y*(-s_z*camera_y+c_z*camera_x) + s_y*camera_z
    d_y = -s_x*(c_y*camera_z-s_y*(-s_z*camera_y+c_z*camera_x))+c_x*(c_z*camera_y+s_z*camera_x)
    d_z = c_x*(c_y*camera_z-s_y*(-s_z*camera_y+c_z*camera_x))+s_x*(c_z*camera_y+s_z*camera_x)
    d = [d_x,d_y,d_z]
    return d
end

function coordinates_to_pixels(f,coordinates_to_camera_pos,image_size,pixels_size)::Point
    k_x = pixels_size[1]/image_size[1]
    k_z = pixels_size[2]/image_size[2]
    # first convert to image space coordinates
    i_x = f/coordinates_to_camera_pos[2]*coordinates_to_camera_pos[1]
    i_y = f
    i_z = f/coordinates_to_camera_pos[2]*coordinates_to_camera_pos[3]
    #println([i_x,i_z])
    # then convert to pixels coordinates
    p_x = k_x*i_x+pixels_size[1]/2
    p_z = k_z*i_z+pixels_size[2]/2
    p = Point(p_x,p_z)
    return p
end


