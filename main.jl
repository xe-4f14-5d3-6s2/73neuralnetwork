using Distributions
using Flux
using JSON3
using CodecZlib


#################### Layer
mutable struct Layer
	W::Matrix{Float64}
	b::Vector{Float64}
	σ::Function
end
function Layer(output::Int64, input::Int64, σ::Function)
	W = rand(output, input)
	b = rand(output)
	return Layer(W, b, σ)
end
function (dns::Layer)(x::Vector{Float64})
	return dns.σ.(dns.W * x + dns.b)
end
#################### End Layer

#################### Model
mutable struct Model
	layers::Vector{Layer}
end
function (chn::Model)(x::Vector{Float64})
	for layer in chn.layers
		x = layer(x)
	end
	return x
end
function save(model::Model, file_name::String)
	data = JSON3.write([ [[row for row in eachrow(layer.W)], layer.b] for layer in model.layers])
	open("$file_name.73nn", "w") do file
		gz = GzipCompressorStream(file)
		write(gz, data)
		close(gz)
	end
	println("Archivo guardado como $file_name.73nn")
	return 0
end
function load(path::String, activaction::Vector)
	data = open(path, "r") do file
		gz = GzipDecompressorStream(file)
		read(gz, String)
	end

	data_array = JSON3.read(data)
	model = Model([Layer(size(layer[1], 1), size(layer[1], 2), activaction[L]) for (L,layer) in enumerate(data_array)])
	for (L, layer) in enumerate(model.layers)
		layer.W = reduce(hcat, [Vector{Float64}(row) for row in data_array[L][1]])'
		layer.b = Vector(data_array[L][2])
	end
	return model
end
#################### End Model


#model = Model([Layer(2, 2, leakyrelu), Layer(2, 2, leakyrelu)])
#println(model([200.0, 500.0]))
#[364.89265605641253, 524.3266738277267]

#save(model, "modelo_ej2")

#model2 = load("modelo_ej2.73nn", [leakyrelu, leakyrelu])
#println(model2([200.0, 500.0]))
#[364.89265605641253, 524.3266738277267]