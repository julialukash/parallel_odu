% [pi, mu, L] = EM(X, options)
% X � ������� ������� N ? D, ����������� ������ X,
% options � ��������� � ������ �a�, �b�, �alpha�, �K�, �max_iter�, �tol�, �n_start�.
% a, b � ��������� ���������� ���� �������������,
% alpha � �������� ���������� ������������� �������,
% K � ����� ��������� K,
% max_iter � ������������ ����� ��������,
% tol � �������� ����������� �� L(q),
% n_start � ����� �������� �� ��������� ��������� ��������� �����������.
% ����� ���������:
% pi � ������ ���������� ?, ������� ������� K ? 1,
% mu � �������� Eq(�)�, ������� ������� K ? D,
% L � �������� ����������� L(q) �� ���������, ��� ������� ���������� �����������.

function [pi, mu, L, iter, R, new_k, L_s] = EM(X, options)
    if  ~isfield(options,'a'), options.a = 1; end
    if  ~isfield(options,'b'), options.b = 1; end
    if  ~isfield(options,'alpha'), options.alpha = 0.001; end
    if  ~isfield(options,'K'), options.K = 50; end
    if  ~isfield(options,'max_iter'), options.max_iter = 500; end
    if  ~isfield(options,'tol'), options.tol = 1e-3; end
    if  ~isfield(options,'n_start'), options.n_start = 1; end
    [N, D] = size(X);
    K = options.K;    
    functionals = zeros(1, options.n_start);
    pi_s = cell(1, options.n_start);
    mu_s = cell(1, options.n_start);
    L_s = mu_s;
    for n_start = 1 : options.n_start
        pi = ones(K, 1) / K;
        R = rand([N, K]);
        R = R ./ repmat(sum(R,2), [1,numel(pi)]);
        L = zeros(1, options.max_iter / 10);
        for iter = 1 : options.max_iter
            % Eq(mu) a', b'
            A = repmat(options.a, numel(pi), D) + R' * X;
            B = repmat(options.b,numel(pi), D) + R' * (1 - X);
            % Eq(Z)   
            sufficient_statistic_mu = psi(A) - psi(A + B); %E(ln(mu))
            sufficient_statistic_1_mu = psi(B) - psi(A + B); %E(ln(1-mu))
            % Eq(z) = R
            RHO = X * sufficient_statistic_mu' + ...
                (1 - X) * sufficient_statistic_1_mu' + repmat(log(pi)', [N, 1]);
            R = exp(RHO);
            R = R ./ repmat(sum(R,2), [1,numel(pi)]);
            
            % M-step: pi
            pi = (sum(R,1)' + options.alpha - 1) / (N + options.K * (options.alpha - 1));
            indexes_zeros = pi <= 0;
            pi(indexes_zeros) = [];
            pi = pi ./ sum(pi);
            R(:,indexes_zeros) = [];
            R = R ./ repmat(sum(R,2), [1,numel(pi)]);
%             if mod(iter, 50) == 0 
%                 disp(['debug: ', num2str(iter)]);
%             end
            if mod(iter, 10) == 0 
                A(indexes_zeros,:) = [];
                B(indexes_zeros,:) = [];
                sufficient_statistic_mu(indexes_zeros,:) = [];
                sufficient_statistic_1_mu(indexes_zeros,:) = [];
                RHO(:,indexes_zeros) = [];
                r_non_zero = R(R > 0);
                L(iter/10) = sum(sum(RHO .* R)) + gammaln(options.K * options.alpha) - ...
                options.K * gammaln(options.alpha) + (options.alpha - 1) * sum(log(pi)) - ...
                options.K * D * betaln(options.a, options.b) + ...
                sum(sum((options.a - 1)*sufficient_statistic_mu + ...
                (options.b - 1)*sufficient_statistic_1_mu)) - sum(sum(r_non_zero.*log(r_non_zero))) - ...
                sum(sum((A - 1) .* sufficient_statistic_mu + (B - 1) .* sufficient_statistic_1_mu)) + ...
                sum(sum(betaln(A, B)));
                if iter > 10 && L(iter/10) - L(iter/10 - 1) < options.tol
                    break;
                end
            end
        end
        functionals(n_start) = L(iter/10);
        pi_s{n_start} = pi;
        mu =  A ./ (A + B);
        mu_s{n_start} = mu;
        L_s{n_start} = L;
    end
    [max_L, index_max] = max(functionals);
    pi = pi_s{index_max};
    mu = mu_s{index_max};
    new_k = numel(pi);
    mu = [mu;  repmat(options.a / (options.a + options.b), options.K - numel(pi), D)];
    pi = [pi; zeros(options.K - numel(pi),1)];
end
    