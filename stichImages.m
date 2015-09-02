function [resultImage, resultMask] = stichImages(images, seeds)
K = numel(images);
[N, M, ~] = size(images{1});
unary = zeros(N, M, K);
tmp = zeros(N, M);
for i = 1 : K
    tmp = tmp | seeds{i};
end
for i = 1 : K
    unary(:, :, i) = xor(tmp, seeds{i});
end
unary(unary ~= 0) = Inf * unary(unary ~= 0);
gray = zeros(N, M, K);
for i = 1 : K
    gray(:, :, i) = rgb2gray(images{i});
end    
difference_gray = max(gray, [], 3) - min(gray, [], 3);
vertC = difference_gray(1 : N - 1, :);
horC = difference_gray(:, 1 : M - 1);
metric = ones(K) - eye(K);
[labels, energy, time] = alphaExpansionGridPotts(unary, vertC, horC, metric);
% [labels, energy, time] = alphaBetaSwapGridPotts_Turin(unary, vertC, horC, metric);
energy
time
resultMask = labels;
resultImage = zeros(N, M, 3);
for i = 1 : N
    for j = 1 : M
        resultImage(i, j, :) = images{labels(i, j)}(i, j, :);
    end
end
resultImage = uint8(resultImage);