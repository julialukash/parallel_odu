im1_rgb = imread('pics/11.jpg');
im2_rgb = imread('pics/12.jpg');
im3_rgb = imread('pics/13.jpg');
im4_rgb = imread('pics/15.jpg');
images = {im1_rgb; im2_rgb; im3_rgb; im4_rgb};
[N, M, ~] = size(im1_rgb);
seed1 = zeros(N, M);
seed1(340 : 350, 150 : 160) = 1;
seed2 = zeros(N, M);
seed2(220 : 230, 540 : 580) = 1;
seed3 = zeros(N, M);
seed3(140 : 150, 320 : 330) = 1;
seed4 = zeros(N, M);
seed4(430 : 440, 450 : 460) = 1;
seeds = {logical(seed1); logical(seed2); logical(seed3); logical(seed4)};
[resultImage, resultMask] = stichImages(images, seeds);
imshow(resultImage);
imwrite(resultImage, 'pics/four_images.jpg')
