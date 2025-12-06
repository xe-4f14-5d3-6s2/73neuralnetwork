using Distributions

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

function Model_generate_layer(n::Int64)
	return fill(Neuron(), n)
end

function Model_generate_network(layers::Vector{Int64})
	return [Model_generate_layer(n) for n in layers]
end

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

function Model(layers::Vector{Int64}, activaction::String)
	m = Model(Model_generate_network(layers), activaction)

	for (L, layer) in enumerate(m.network)
		if L != 1
			for neuron in layer
				neuron.x = [Connection(n) for n in m.network[L-1]]
			end
		end
	end
	return m

end
#################### End Model