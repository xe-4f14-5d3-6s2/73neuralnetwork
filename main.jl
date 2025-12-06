using Distributions
using JSON3
using CodecZlib

############## Neuron
mutable struct Neuron
	x::Vector
	a::Float64
	z::Float64
	bias::Float64
end
Neuron() = Neuron([], 0.0, 0.0, rand(Uniform(-1.0, 1.0)))
############## End Neuron

######################### Connection
mutable struct Connection
	neuron::Neuron
	weight::Float64
end
Connection(neuron::Neuron) = Connection(neuron, rand(Uniform(-1.0, 1.0)))
######################### End Connection

#################### Model
mutable struct Model
	network::Array{Array{Neuron, 1},1}
	activaction::String
end

generate_layer(n::Int64) = fill(Neuron(), n)

generate_network(layers::Vector{Int64}) = [generate_layer(n) for n in layers]

function activaction(model::Model, z::Vector{Float64})
	if model.activaction == "Sigmoid"
		return 1.0 ./ (1.0 .+ exp.(-z))
	elseif model.activaction == "ReLU"
		return max.(0.0, z)
	elseif model.activaction == "LeakyReLU"
		return [x >= 0 ? x : 0.01 * x for x in z]
	end
	return max.(0.0, z)
end

function predict(model::Model, data::Vector{Float64})
	if length(data) == length(model.network[1])
		for (V, value) in enumerate(data)
			model.network[1][V].a = value
		end
		for (L, layer) in enumerate(model.network[2:length(model.network)])
			for (N, neuron) in enumerate(layer)
				z = Float64(sum([con.neuron.a * con.weight for con in neuron.x]) + neuron.bias)
				a = activaction(model, [z])[1]
				neuron.z = z
				neuron.a = a
			end
		end

		return model.network[length(model.network)]
	end
end

function is_valid_73nn(path::String)
    endswith(path, ".73nn") || return false
    isfile(path) || return false
    data_string = try
        open(path, "r") do f
            String(read(GzipDecompressorStream(f)))
        end
    catch
        return false
    end
    data = try
        JSON3.read(data_string)
    catch
        return false
    end
    isa(data, AbstractVector) || return false
    for layer in data
        isa(layer, AbstractVector) || return false
        for neuron in layer
            isa(neuron, AbstractVector) || return false
            length(neuron) == 2 || return false

            bias = neuron[1]
            weights = neuron[2]

            (isa(bias, Number)) || return false
            isa(weights, AbstractVector) || return false
            all(w -> isa(w, Number), weights) || return false
        end
    end
    return true
end

function save(model::Model, file_name::String)
	data = JSON3.write([[ [neuron.bias, [conn.weight for conn in neuron.x]] for neuron in layer ] for layer in model.network])
	open("$file_name.73nn", "w") do file
		gz = GzipCompressorStream(file)
		write(gz, data)
		close(gz)
	end
	println("Archivo guardado como $file_name.73nn")
	return 0
end

function Model(layers::Vector{Int64}, activaction::String)
	m = Model(generate_network(layers), activaction)

	for (L, layer) in enumerate(m.network)
		if L != 1
			for neuron in layer
				neuron.x = [Connection(n) for n in m.network[L-1]]
			end
		end
	end
	return m

end

function load(path::String, activaction::String)
	if is_valid_73nn(path)
		data = open(path, "r") do file
			gz = GzipDecompressorStream(file)
			read(gz, String)
		end

		data_array = JSON3.read(data)

		model = Model([length(layer) for layer in data_array], activaction)

		for (L, layer) in enumerate(model.network)
			for (N, neuron) in enumerate(layer)
				neuron.bias = data_array[L][N][1]
				for (C, conn) in enumerate(neuron.x)
					conn.weight = data_array[L][N][2][C]
				end
			end
		end

		return model
	end
	return 0
end
#################### End Model

#model = Model([2,4,4,1], "Sigmoid")
#model = load("modelo_ej1.73nn", "Sigmoid")

#println(predict(model, [5.0,2.0])[1].a)
#println(predict(model, [10.0,20.0])[1].a)

#0.771367836758079
#0.7482614929538031

#save(model, "modelo_ej1")