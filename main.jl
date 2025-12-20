using Distributions, Random, Flux, JSON3, CodecZlib, Zygote


#################### Layer
mutable struct Layer
	W::Matrix{Float64}
	b::Vector{Float64}
	z::Vector{Float64}
	σ::Function
	dσ::Function
end
function Layer(output::Int64, input::Int64, σ::Function)
	W = randn(output, input) .* 0.1
	b = rand(output)
	z = rand(output)
	dσ(x) = Zygote.gradient(σ, x)[1]
	return Layer(W, b, z, σ, dσ)
end
function (dns::Layer)(x::Vector{Float64})
	dns.z = dns.W * x + dns.b
	return dns.σ.(dns.z)
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
function fit(model::Model, dataset::Vector{Vector{Vector{Float64}}}, epochs::Int64, lr_w::Float64, lr_b::Float64)
	for ep in 1:epochs
		println("Epoch: $ep")
		for (X, y) in shuffle(dataset)
			ŷ = model(X)
		
			δ = []

			ðC_ðaL = 2 * (ŷ - y)
			ðaL_ðzL = model.layers[end].dσ.(model.layers[end].z)
			δL = ðC_ðaL .* ðaL_ðzL
			pushfirst!(δ, δL)

			for l in (length(model.layers)-1):-1:1
				ðal_ðzl = model.layers[l].dσ.(model.layers[l].z)
				backprop_err = model.layers[l+1].W' * δ[1]

				δl = ðal_ðzl .* backprop_err

				pushfirst!(δ, δl)
			end

			acts = [X]
			for l in model.layers
				push!(acts, l.σ.(l.z))
			end

			for l in 1:length(model.layers)
				ðC_ðWl = δ[l] * acts[l]'
				model.layers[l].W .-= lr_w .* ðC_ðWl
				model.layers[l].b .-= lr_b .* δ[l]
			end
		end
	end
end
#################### End Model

max_input = 5000.0
max_output = (5000.0 * 1.8) + 32.0

#dataset = [[[x/max_input],[(x * 1.8 + 32)/max_output]] for x in 1:5000]

#model = Model([
#	Layer(1,1, leakyrelu),
#	Layer(1,1, x -> x)
#])

#fit(model, dataset, 500, 0.01, 0.01)

test_val = 10.0

#pred_norm = model([test_val / max_input])[1]

#println("Predicción: ", pred_norm * max_output)
#save(model, "celsius_to_fahrenheit")
model2 = load("celsius_to_fahrenheit.73nn", [leakyrelu, x -> x])
pred_norm2 = model2([test_val / max_input])[1]
println("Predicción 2: ", pred_norm2 * max_output)
#model = Model([Layer(2, 2, leakyrelu), Layer(2, 2, leakyrelu)])
#println(model([200.0, 500.0]))
#[364.89265605641253, 524.3266738277267]

#save(model, "modelo_ej2")

#model2 = load("modelo_ej2.73nn", [leakyrelu, leakyrelu])
#println(model2([200.0, 500.0]))
#[364.89265605641253, 524.3266738277267]
