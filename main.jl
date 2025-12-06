using Distributions

############# Neuron
mutable struct Neuron
	x::Vector
	a::Float64
	z::Float64
	bias::Float64
end
Neuron() = Neuron([], 0.0, 0.0, rand(Uniform(-1.0, 1.0)))
############# End Neuron

################# Connection
mutable struct Connection
	neuron::Neuron
	weight::Float64
end
Connection(neuron::Neuron) = Connection(neuron, rand(Uniform(-1.0, 1.0)))
################# End Connection

############ Model
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
############ End Model