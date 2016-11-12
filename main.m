input_folder = 'output_lomo/output';
% [mat_i, res_i] = gather_matrix(block, fullfile(input_folder, 'init'),  'init_rank', 1);
[mat_f, res_f] = gather_matrix(block, fullfile(input_folder, 'finish'), ['p_', num2str(ms(1)), 'x', num2str(ms(2)), '_fin_rank'], 0);
[mat_u, res_u] = gather_matrix(block, fullfile(input_folder, 'true'), ['u_', num2str(ms(1)), 'x', num2str(ms(2)), '_rank'], 0);