im1_rgb = imread('pics/1.png');
im2_rgb = imread('pics/2.png');
images = {im1_rgb; im2_rgb};
[N, M, ~] = size(im1_rgb);
seed1 = zeros(N, M);
seed1(20 : 30, 70 : 80) = 1;
seed2 = zeros(N, M);
seed2(20 : 30, 20 : 30) = 1;
seeds = {logical(seed1); logical(seed2)};
[resultImage, resultMask] = stichImages(images, seeds);
imwrite(resultImage, 'pics/model_images.jpg')