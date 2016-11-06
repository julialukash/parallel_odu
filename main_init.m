x_block = block(1);
y_block = block(2);
folder = 'init';
name = 'init_rank';
mat_i = cell(x_block, y_block);
it = 0;
for i = 1 : x_block
    for j = 1 : y_block
        rank = it;
        c_name = [name, int2str(rank), '.txt'];
        k = csvread(fullfile(folder, c_name));
        mat_i{i, j} = k;
        it = it + 1;
    end
end
mat_i = mat_i';
mat_i = flipud(mat_i);
delete = 1;
if delete
    for i = 1 : numel(mat_i)
        tmp = mat_i{i};
        [s1, s2] = size(tmp);
        mat_i{i} = tmp(2 : s1 - 1, 2 : s2 - 1);
    end
end
clear it c_name k rank folder name x_block y_block i j delete tmp s1 s2
res_i = cell2mat(mat_i);