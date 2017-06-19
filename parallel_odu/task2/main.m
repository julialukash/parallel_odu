input_folder = 'output_lomo/src_parallel1/output';
% [mat_i, res_i] = gather_matrix(block, fullfile(input_folder, 'init'),  'init_rank', 1);
[mat_f, res_f] = gather_matrix(block, fullfile(input_folder, 'finish'), ['p_', num2str(ms(1)), 'x', num2str(ms(2)), '_', num2str(processorsCount) ,'__rank'], 0);
[mat_u, res_u] = gather_matrix(block, fullfile(input_folder, 'true'), ['u_', num2str(ms(1)), 'x', num2str(ms(2)), '_', num2str(processorsCount) , '__rank'], 0);