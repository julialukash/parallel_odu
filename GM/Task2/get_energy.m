function energy = get_energy(labels, unary, vertC, horC, metric)
[N, M, ~] = size(unary);
energy = 0;
for i = 1 : N
    for j = 1 : M
        energy = energy + unary(i, j, labels(i, j));
    end
end

for i = 1 : N - 1
    for j = 1 : M
        energy = energy + vertC(i, j) * metric(labels(i, j), labels(i + 1, j));
    end
end


for i = 1 : N 
    for j = 1 : M - 1
        energy = energy + horC(i, j) * metric(labels(i, j), labels(i, j + 1));
    end
end
