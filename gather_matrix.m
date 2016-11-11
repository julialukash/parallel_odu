function [mat, res] = gather_matrix(block, folder, name, delete)
    x_block = block(1);
    y_block = block(2);
    mat = cell(x_block, y_block);
    it = 0;
    for i = 1 : x_block
        for j = 1 : y_block
            rank = it;
            c_name = [name, int2str(rank), '.txt'];
            k = csvread(fullfile(folder, c_name));
            mat{i, j} = k;
            it = it + 1;
        end
    end
    mat = mat';
    mat = flipud(mat);
    if delete
        for i = 1 : numel(mat)
            tmp = mat{i};
            [s1, s2] = size(tmp);
            mat{i} = tmp(2 : s1 - 1, 2 : s2 - 1);
        end
    end
    res = cell2mat(mat);