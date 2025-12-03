w = 0.1
b = 0
uw = 0.00000001
ub = 0.001

training_data = [(c, c*3.281) for c in range(1000)]

for epoch in range(500):
    for cel, far in training_data:
        y = cel*w + b
        e = far - y
        w += uw * e * cel
        b += ub * e

print(round(w,3), round(b,3))