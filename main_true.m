x_block = block(1);
y_block = block(2);
folder = 'true';
name = 'u_rank';
mat_u = cell(x_block, y_block);
it = 0;
for i = 1 : x_block
    for j = 1 : y_block
        rank = it;
        c_name = [name, int2str(rank), '.txt'];
        k = csvread(fullfile(folder, c_name));
        mat_u{i, j} = k;
        it = it + 1;
    end
end
mat_u = mat_u';
mat_u = flipud(mat_u);
delete = 0;
if delete
    for i = 1 : numel(mat_u)
        tmp = mat_u{i};
        [s1, s2] = size(tmp);
        mat_u{i} = tmp(2 : s1 - 1, 2 : s2 - 1);
    end
end
clear it c_name k rank folder name x_block y_block i j delete tmp s1 s2
res_u = cell2mat(mat_u);