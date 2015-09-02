function [labels, energy, time] = alphaBetaSwapGridPotts(unary, vertC, horC, metric, varargin)

maxIter = 500;
display = false;
numStart = 1;
randOrder = false;
if ~isempty(varargin)
    options = varargin{1};
    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    end
    if isfield(options, 'display')
        display = options.display;
    end
    if isfield(options, 'numStart')
        numStart = options.numStart;
    end
    if isfield(options, 'randOrder')
        randOrder = options.randOrder;
    end
end

N = size(unary, 1);
M = size(unary, 2);
K = size(unary, 3);

if K == 2
    tic;
    %     [vertC, horC] = reparametrization(vertC, horC, metric);
    vertC = metric(1,2) * vertC;
    horC = metric(1,2) * horC;
    termWeights = reshape(unary,[size(unary,1)*size(unary,2) 2]);
    edgeWeights = zeros(N*(M - 1) + (N - 1)*M,4);
    horC = horC';
    edgeWeights(1:N*(M - 1),3) = horC(:);
    horC = horC';
    edgeWeights(1:N*(M - 1),4) = edgeWeights(1:N*(M - 1),3);
    index = 1:N;
    index = repmat(index,[M - 1, 1]);
    index = index(:) + repmat(0:N:(N*(M-2)),[1 N])';
    edgeWeights(1:N*(M - 1),1) = index;
    edgeWeights(1:N*(M - 1),2) = index + N;
    edgeWeights(N*(M - 1)+1:end,3) = vertC(:);
    edgeWeights(N*(M - 1)+1:end,4) = edgeWeights(N*(M - 1)+1:end,3);
    index = 1:N;
    index = [index(1:end-1);index(2:end)]';
    index = repmat(index,[M 1]);
    indexTemp = 0:N:(M - 1)*N;
    indexTemp = repmat(indexTemp, [2*(N - 1) 1]);
    index = index + reshape(indexTemp, [2 (N - 1)*M])';
    edgeWeights(N*(M - 1)+1:end,1:2) = index;
    [energy, labels] = graphCutMex(termWeights, edgeWeights);
    labels = reshape(labels, [N M]);
    temp = labels;
    labels(temp == 0) = 2;
    labels(temp == 1) = 1;
    %     energy_ = countEnergy(unary, vertC, horC, metric, labels);
    time = toc;
    return;
end

alphaBetaIndex = zeros(K*(K-1)/2,2);
p = 0;
for i = 1:K
    for j = i+1:K
        p = p + 1;
        alphaBetaIndex(p,:) = [i,j];
    end
end

for start = 1:numStart
    
    tic;
    
    labels = zeros(N, M);
    labels(:) = randi([1 K], N*M, 1);
%     labels(:) = 1;
    
    for iter = 1 : maxIter
        success = false;
        if randOrder
            randOrderTemp = randperm(K*(K-1)/2);
            alphaBetaIndex = alphaBetaIndex(randOrderTemp,:);
        end
        for pAll = 1:K*(K-1)/2
            alpha = alphaBetaIndex(pAll, 1);
            beta = alphaBetaIndex(pAll, 2);
            maskAlpha = labels == alpha;
            maskBeta = labels == beta;
            maskAll = maskAlpha | maskBeta;
            if ~sum(maskAll(:))
                continue;
            end
            maskHor = (maskAll(:,1:end-1) == maskAll(:,2:end))&(maskAll(:,1:end-1) == 1);
            maskVert = (maskAll(1:end-1,:) == maskAll(2:end,:))&(maskAll(1:end-1,:) == 1);
            termWeights = zeros(sum(maskAll(:)),2);
            edgeWeights = zeros(sum(maskHor(:)) + sum(maskVert(:)),4);
            pIndex = zeros(N, M);
            IndexP = zeros(sum(maskAll(:)), 2);
            p = 0;
            for i = 1:N
                for j = 1:M
                    if ~maskAll(i,j)
                        continue;
                    end
                    p = p + 1;
                    pIndex(i, j) = p;
                    IndexP(p,:) = [i, j];
                    termWeights(p, 1) = unary(i,j,alpha);
                    termWeights(p, 2) = unary(i,j,beta);
                    if (j >= 2) && ~maskAll(i,j - 1)
                        termWeights(p, 1) = termWeights(p, 1) +...
                            horC(i,j - 1) * metric(alpha, labels(i, j - 1));
                        termWeights(p, 2) = termWeights(p, 2) +...
                            horC(i,j - 1) * metric(beta, labels(i, j - 1));
                    end
                    if (j <= M - 1) && ~maskAll(i,j + 1)
                        termWeights(p, 1) = termWeights(p, 1) +...
                            horC(i,j) * metric(alpha, labels(i, j + 1));
                        termWeights(p, 2) = termWeights(p, 2) +...
                            horC(i,j) * metric(beta, labels(i, j + 1));
                    end
                    if (i >= 2) && ~maskAll(i - 1,j)
                        termWeights(p, 1) = termWeights(p, 1) +...
                            vertC(i - 1,j) * metric(alpha, labels(i - 1, j));
                        termWeights(p, 2) = termWeights(p, 2) +...
                            vertC(i - 1,j) * metric(beta, labels(i - 1, j));
                    end
                    if (i <= N - 1) && ~maskAll(i + 1,j)
                        termWeights(p, 1) = termWeights(p, 1) +...
                            vertC(i,j) * metric(alpha, labels(i + 1, j));
                        termWeights(p, 2) = termWeights(p, 2) +...
                            vertC(i,j) * metric(beta, labels(i + 1, j));
                    end
                end
            end
            p = 0;
            for i = 1:N
                for j = 1:M - 1
                    if maskHor(i, j)
                        p = p + 1;
                        edgeWeights(p,:) = [pIndex(i,j) pIndex(i,j+1)...
                            metric(alpha, beta) * horC(i,j) metric(alpha, beta) * horC(i,j)];
                    end
                end
            end
            
            for i = 1:N-1
                for j = 1:M
                    if maskVert(i, j)
                        p = p + 1;
                        edgeWeights(p,:) = [pIndex(i,j) pIndex(i+1,j)...
                            metric(alpha, beta) * vertC(i,j) metric(alpha, beta) * vertC(i,j)];
                    end
                end
            end
            
            [en, labelsAlphaBeta] = graphCutMex(termWeights, edgeWeights);
            
            energy_ = countEnergy(unary, vertC, horC, metric, labels);
            if (iter == 1) || (abs(energy_ - energy(iter - 1)) > 1e-6)||(isnan(energy_ - energy(iter - 1)))
                indexTemp = labelsAlphaBeta == 0;
                labels(sub2ind([N M],IndexP(indexTemp,1),IndexP(indexTemp,2))) = beta;
                indexTemp = labelsAlphaBeta == 1;
                labels(sub2ind([N M],IndexP(indexTemp,1),IndexP(indexTemp,2))) = alpha;
                success = true;
            end
            if true
                disp([' ']);
                disp(['---------------------------']);
                disp(['Iteration: ', num2str(iter)]);
                disp(['Alpha - Beta: ', num2str(alpha),' - ', num2str(beta)]);
                disp(['Energy: ', num2str(energy_)]);
                disp(['---------------------------']);
                disp([' ']);
            end
        end
        energy(iter) = countEnergy(unary, vertC, horC, metric, labels);
        time(iter) = toc;
        if ~success
            break;
        end
        
    end
    
    if start == 1
        labelsBest = labels;
        energyBest = energy;
        timeBest = time;
    end
    
    if (start > 1) && (energy(end) < energyBest(end))
        labelsBest = labels;
        energyBest = energy;
        timeBest = time;
    end
    
end

labels = labelsBest;
energy = energyBest;
time = timeBest;

end

function E = countEnergy(unary, vertC, horC, metric, labels)
E = 0;
N = size(unary, 1);
M = size(unary, 2);
for i = 1:N
    for j = 1:M
        E = E + unary(i,j,labels(i,j));
    end
end

for i = 1:N
    for j = 1:M-1
        E = E + horC(i,j)*metric(labels(i,j), labels(i,j+1));
    end
end

for i = 1:N-1
    for j = 1:M
        E = E + vertC(i,j)*metric(labels(i,j), labels(i+1,j));
    end
end
end

