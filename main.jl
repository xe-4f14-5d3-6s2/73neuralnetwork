using Distributions

struct Neuron
	x::Vector
	a::Float32
	z::Float32
	bias::Float32
end
Neuron() = Neuron([], 0.0, 0.0, rand(Uniform(-1.0, 1.0)))

struct Connection
	neuron::Neuron
	weight::Float32
end
Connection(neuron::Neuron) = Connection(neuron, rand(Uniform(-1.0, 1.0)))

struct Model
	network::Matrix{Neuron}
	activaction::String
end