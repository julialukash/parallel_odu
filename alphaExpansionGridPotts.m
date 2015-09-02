function [labels_best, energy_best, time_best] = alphaExpansionGridPotts(unary, vertC, horC, metric, options)
if ~exist('options', 'var')
    options = struct();
end    
if ~isfield(options, 'maxIter')
        options.maxIter = 500;
end
if ~isfield(options, 'numStart')
    options.numStart = 1;
end
if ~isfield(options, 'randOrder')
    options.randOrder = false;
end
if ~isfield(options, 'display')
    options.display = true;
end
[N, M, K] = size(unary);
energy_best = Inf;
time_best = [];
labels_best = [];
ks = 1 : K;
for num_start = 1 : options.numStart
    tic;
    energy = zeros(1, options.maxIter);
    time = zeros(1, options.maxIter);
    labels = randi([1, K], [N, M]);
    for iter = 1 : options.maxIter
        if options.randOrder
            ks = randperm(K);
        end
        for k = 1 : K
            alpha = ks(k);
            psi_0 = zeros(N, M);
            for i = 1 : N
                for j = 1 : M
                    psi_0(i, j) = unary(i, j, labels(i, j));  
                end
            end 
            psi_1 = unary(:, :, alpha);
            vert_i_j_0_0 = zeros(N - 1, M);
            vert_i_j_0_1 = zeros(N - 1, M);
            vert_i_j_1_0 = zeros(N - 1, M);
            vert_i_j_1_1 = zeros(N - 1, M);
            
            hor_i_j_0_0 = zeros(N, M - 1);
            hor_i_j_0_1 = zeros(N, M - 1);
            hor_i_j_1_0 = zeros(N, M - 1);
            hor_i_j_1_1 = zeros(N, M - 1);
            
            for i = 1 : N - 1
                for j = 1 : M
                    vert_i_j_0_0(i, j) = vertC(i, j) * metric(labels(i, j), labels(i + 1, j));
                    vert_i_j_0_1(i, j) = vertC(i, j) * metric(labels(i, j), alpha);                                        
                    vert_i_j_1_0(i, j) = vertC(i, j) * metric(alpha, labels(i + 1, j));
                    vert_i_j_1_1(i, j) = vertC(i, j) * metric(alpha, alpha);
                end
            end
            for i = 1 : N 
                for j = 1 : M - 1
                    hor_i_j_0_0(i, j) = horC(i, j) * metric(labels(i, j), labels(i, j + 1));
                    hor_i_j_0_1(i, j) = horC(i, j) * metric(labels(i, j), alpha);                                        
                    hor_i_j_1_0(i, j) = horC(i, j) * metric(alpha, labels(i, j + 1));
                    hor_i_j_1_1(i, j) = horC(i, j) * metric(alpha, alpha);
                end
            end
            a = vert_i_j_0_0;
            b = vert_i_j_1_1;
            c = vert_i_j_0_1;
            d = vert_i_j_1_0;
            vert_i_j_0_1 = 0 * vert_i_j_0_1;
            vert_i_j_1_0 = c + d - a - b;

            psi_0(1 : N - 1, :) = psi_0(1 : N - 1, :) + a; % psi_i(0)
            psi_1(2 : N, :) = psi_1(2 : N, :) + c - a; % psi_i(1)
            psi_1(1 : N - 1, :) = psi_1(1 : N - 1, :) + b - c + a; % psi_j(1)

            a = hor_i_j_0_0;
            b = hor_i_j_1_1;
            c = hor_i_j_0_1;
            d = hor_i_j_1_0;
            hor_i_j_0_1 = 0 * hor_i_j_0_1;
            hor_i_j_1_0 = c + d - a - b;

            psi_0(:, 1 : M - 1) = psi_0(:, 1 : M - 1) + a; % psi_i(0)
            psi_1(:, 2 : M) = psi_1(:, 2 : M) + c - a; % psi_j(1)
            psi_1(:, 1 : M - 1) = psi_1(:, 1 : M - 1) + b - c + a; % psi_i(1)

            delta = min(psi_0, psi_1);
            psi_0 = psi_0 - delta;
            psi_1 = psi_1 - delta;

            v = reshape(1 : M * N, [N, M]);
            edgeWeights = zeros(numel(hor_i_j_0_1) + numel(vert_i_j_0_1), 4);
            % edgeWeights(i, 3) connects node #edgeWeights(i, 1) to node #edgeWeights(i, 2)
            % edgeWeights(i, 4) connects node #edgeWeights(i, 2) to node #edgeWeights(i, 1)
            edgeWeights(:, 1) = [reshape(v(1 : N - 1, :), [], 1); reshape(v(:, 1 : M - 1), [], 1)];
            edgeWeights(:, 2) = [reshape(v(2 : N, :), [], 1); reshape(v(:, 2 : M), [], 1)];
            edgeWeights(:, 3) = [vert_i_j_0_1(:); hor_i_j_0_1(:)];
            edgeWeights(:, 4) = [vert_i_j_1_0(:); hor_i_j_1_0(:)];            
            % termWeights(i, 1) is the weight of the edge connecting the
            % source with node #i \psi(1)
            % termWeights(i, 2) is the weight of the edge connecting node
            % #i with the sink \psi (0)
            terminalWeights = [psi_1(:), psi_0(:)];
            [~, new_labels] = graphCutMex(terminalWeights, edgeWeights);
            labels(logical(new_labels)) = alpha;
            labels = reshape(labels, N, M);
            if options.display
                fprintf('Старт: %d  из %d\n', num_start, options.numStart);
                fprintf('Итерация: %d  из %d\n', iter, options.maxIter);
                fprintf('Метка: %d  из %d\n', k, K);
                fprintf('Текущее значение энергии: %d\n', get_energy(labels, unary, vertC, horC, metric));
            end
        end
        energy(iter) =  get_energy(labels, unary, vertC, horC, metric);
        time(iter) = toc;
        if iter ~= 1 && energy(iter - 1) - energy(iter) <= 1e-4;
            break;
        end
    end
    energy = energy(1 : iter);
    time = time(1 : iter);
    if energy(end) <= energy_best(end)
        energy_best = energy;
        time_best = time;
        labels_best = labels;
    end
end
end

