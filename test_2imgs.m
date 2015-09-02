im1_rgb = imread('pics/funtic_fotomodel_1.jpg');
im2_rgb = imread('pics/funtic_fotomodel_2.jpg');
im1 = rgb2gray(im1_rgb);
im2 = rgb2gray(im2_rgb);
images = {im1_rgb; im2_rgb};
[N, M, ~] = size(im1_rgb);
seed1 = zeros(N, M);
seed1(160 : 200, 121 : 200) = 1;
seed2 = zeros(N, M);
seed2(180 : 200, 540 : 600) = 1;
seeds = {logical(seed1); logical(seed2)};
[resultImage, resultMask] = stichImages(images, seeds);
% imshow(resultImage);
imwrite(resultImage, 'pics/two_images.jpg')
