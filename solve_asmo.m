function [z, fval, bPlus, bStar] = solve_asmo(fv, fvStar, lbl, C, gamma, sgmPlus, sgmStar, opts)
% solve_asmo - SMO-style solver (aSMO) for SVM+ using I1, I2, and I3 irreducible sets.
% Based on the Learning Using Privileged Information (LUPI) framework.
% Author: JOSE DANIEL FERNANDEZ - dfernande132@alumno.uned.es

    % --- Preprocessing ---
    % Compute initial kernel matrices
    KMatrPlus = exp(-pdist2(fv, fv).^2 / (2 * sgmPlus^2));
    KMatrStar = exp(-pdist2(fvStar, fvStar).^2 / (2 * sgmStar^2));
    
    % Feature normalization
    fv = normalize(fv);
    fvStar = normalize(fvStar);
    N = size(fv, 1);
    y = lbl(:);

    % --- Normalized Gaussian Kernels ---
    K = exp(-pdist2(fv, fv).^2 / (2 * sgmPlus^2));
    Kstar = exp(-pdist2(fvStar, fvStar).^2 / (2 * sgmStar^2));

    % --- Strict Initialization (as proposed in the paper) ---
    % Initially, all slack penalties are absorbed by the privileged space.
    alpha = zeros(N,1);
    beta  = C * ones(N,1);

    % --- Parameters and structures ---
    iter = 0;
    if nargin < 8, opts = struct(); end
    if ~isfield(opts, 'tol'), opts.tol = 1e-3; end
    if ~isfield(opts, 'maxIter'), opts.maxIter = 1000; end
    if ~isfield(opts, 'verbosity'), opts.verbosity = 0; end % Console logging
    if ~isfield(opts, 'kappa'), opts.kappa = 1e-6; end     % Minimum allowed step size

    % --- Initial Gradients ---
    F_terms = K * (alpha .* y);           % F_terms(i) = sum_j K(i,j) * alpha(j) * y(j)
    f_terms = Kstar * (alpha + beta - C); % f_terms(i) = sum_j Kstar(i,j) * (alpha(j)+beta(j)-C)

    % Dual objective gradients with respect to alpha and beta
    grad_alpha = ones(N,1) - y .* F_terms - (1/gamma) * f_terms;
    grad_beta  = -(1/gamma) * f_terms;

    % --- Main Optimization Loop ---
    while iter < opts.maxIter
        iter = iter + 1;
        if opts.verbosity > 0
            fprintf('Iteration=%d. ', iter);
        end

        % Search for the best feasible direction among the three irreducible set types
        [u1, l1, d1] = direccion_I1_fast(beta, grad_beta, Kstar, gamma, opts.tol, opts.kappa, opts.verbosity);
        [u2, l2, d2] = direccion_I2_fast(alpha, y, grad_alpha, K, Kstar, gamma, opts.tol, opts.kappa, opts.verbosity);
        [u3, l3, d3] = direccion_I3_fast(alpha, beta, y, grad_alpha, grad_beta, K, Kstar, gamma, opts.tol, opts.kappa, opts.verbosity);

        % Working Set Selection: Choose the direction with the maximum objective gain (delta)
        [delta_max, type] = max([d1, d2, d3]);

        % Convergence check: Stop if no significant improvement is found
        if delta_max <= opts.tol
            break;
        end

        switch type
            case 1
                u = u1; lambda_opt = l1;
            case 2
                u = u2; lambda_opt = l2;
            case 3
                u = u3; lambda_opt = l3;
        end

        % --- Incremental Gradient Update ---
        % Instead of recomputing the full O(N^2) gradient, we update it based on the 
        % sparse changes (at most 3 indices) in the dual variables.
        
        % Calculate effective changes in alpha and beta for this iteration
        delta_alpha_update = lambda_opt * u(1:N);       % Sparse vector (N x 1)
        delta_beta_update  = lambda_opt * u(N+1:2*N);   % Sparse vector (N x 1)

        % Update dual variables for the next iteration
        alpha = alpha + delta_alpha_update;
        beta  = beta  + delta_beta_update;
        
        % Step 1: Compute change in F_terms (decision space)
        % Only affected by changes in delta_alpha
        delta_F_terms = zeros(N,1);
        indices_alpha_changed = find(delta_alpha_update ~= 0);
        for idx_j = indices_alpha_changed' 
            delta_F_terms = delta_F_terms + K(:, idx_j) * (y(idx_j) * delta_alpha_update(idx_j));
        end

        % Step 2: Compute change in f_terms (correction/privileged space)
        % Affected by both alpha and beta updates
        delta_f_terms = zeros(N,1);
        % Contribution from updated alphas
        for idx_j = indices_alpha_changed'
            delta_f_terms = delta_f_terms + Kstar(:, idx_j) * delta_alpha_update(idx_j);
        end
        % Contribution from updated betas
        indices_beta_changed = find(delta_beta_update ~= 0);
        for idx_j = indices_beta_changed'
            delta_f_terms = delta_f_terms + Kstar(:, idx_j) * delta_beta_update(idx_j);
        end
        
        % Step 3: Apply incremental updates to the gradient vectors
        grad_alpha = grad_alpha - y .* delta_F_terms - (1/gamma) * delta_f_terms;
        grad_beta  = grad_beta - (1/gamma) * delta_f_terms;
        
        % Constraint violation check (debugging)
        if opts.verbosity > 0
            if abs(sum( y .* alpha)) > 0.001
              fprintf('Constraint violation: [%d] at iteration %d\n', sum( y .* alpha), iter);
            end
        end
    end

    % --- Bias Term (b) Calculation ---
    % Derived from KKT conditions on support vectors (alpha_i > 0)
    idxDisruptiveFV = find(y < 0);
    idxNonDisruptiveFV = find(y > 0);

    n_pos = length(idxNonDisruptiveFV);
    n_neg = length(idxDisruptiveFV);

    % Final state of kernel-multiplier products
    F = KMatrPlus * (alpha .* lbl);
    f = KMatrStar * (alpha + beta - C);

    s_pos = sum(1 - f(idxNonDisruptiveFV)/gamma - F(idxNonDisruptiveFV));
    s_neg = sum(-1 + f(idxDisruptiveFV)/gamma - F(idxDisruptiveFV));

    bPlus = 0.5 * (s_pos/n_pos + s_neg/n_neg);
    bStar = 0.5 * (s_pos/n_pos - s_neg/n_neg);

    % --- Output Assignment ---
    z = [alpha; beta];
    fval = sum(alpha) - 0.5 * (alpha .* y)' * K * (alpha .* y) - 0.5 / gamma * (alpha + beta - C)' * Kstar * (alpha + beta - C);

    if opts.verbosity > 0
        fprintf('aSMO terminated in %d iterations. fval = %.6f\n', iter, fval);
    end
end


%% ===================== O(n) Direction Search Functions =====================

% ======================= Type I1 Direction =======================
function [u_best, lambda_best, delta_best] = direccion_I1_fast(beta, grad_beta, Kstar, gamma, tol, kappa, verbosity)
    % direccion_I1_fast: Finds the best Type I1 search direction.
    % Updates two beta variables (beta_s1, beta_s2) to preserve the balance constraint.
    % Direction u_s: (u_s)_beta_s1 = +1, (u_s)_beta_s2 = -1.
    % This preserves sum(alpha_i + beta_i - C) = 0.

    N = length(beta);

    % --- WSS STEP 1: Pre-selection of the first index (i_star) ---
    % Select the index with the largest gradient component to maximize potential ascent.
    [~, i_star] = max(grad_beta);

    u_best      = [];
    lambda_best = 0;
    delta_best  = 0; % Stores maximum objective improvement (delta D)

    % --- WSS STEP 2: Refinement - Linear search for the second index (j) ---
    for j = 1:N
        % Feasibility check: j cannot be i_star, and beta(j) must be > 0 to be decreased.
        if j == i_star || beta(j) <= 1e-7
            continue;
        end

        % Directional gradient: g^T u = g_beta(i_star) - g_beta(j)
        grad_dot = grad_beta(i_star) - grad_beta(j);

        % Tau condition: only proceed if the directional gradient exceeds tolerance.
        if grad_dot <= tol
            continue;
        end

        % Denominator (Curvature): u^T (-H_D) u
        % For Type I1, denom = (1/gamma) * (K*(i*,i*) + K*(j,j) - 2*K*(i*,j))
        denom = (Kstar(i_star,i_star) + Kstar(j,j) - 2*Kstar(i_star,j)) / gamma;

        if denom <= 1e-12
            continue;
        end

        % Unconstrained optimal step (lambda_raw)
        lambda_raw = grad_dot/denom;

        % Box constraint clipping: beta_j_new = beta_j_old - lambda >= 0
        lambda_clip = min(lambda_raw, beta(j));
        lambda_clip = max(0, lambda_clip);

        % Kappa condition: the step must be significantly larger than zero.
        if lambda_clip < kappa
            continue;
        end

        % Expected gain in the dual objective D
        delta_D = lambda_clip*grad_dot - 0.5*lambda_clip^2*denom;

        if delta_D > delta_best
            delta_best  = delta_D;
            lambda_best = lambda_clip;
            u_best      = zeros(2*N,1);
            u_best(N+i_star) =  1;
            u_best(N+j)      = -1;
        end
    end

    if verbosity && ~isempty(u_best)
        fprintf('[I1-fast] i=%d lambda=%.3e deltaD=%.3e', i_star, lambda_best, delta_best);
    end
end

% ======================= Type I2 Direction =======================
function [u_best, lambda_best, delta_best] = direccion_I2_fast(alpha, y, grad_alpha, K, Kstar, gamma, tol, kappa, verbosity)
    % direccion_I2_fast: Finds the best Type I2 direction (two alpha variables of the same class).
    
    N = length(alpha);
    pos_mask = (y ==  1); 
    neg_mask = (y == -1); 

    % WSS STEP 1: Pre-selection - Select index with largest gradient in each class
    grad_alpha_pos_class = grad_alpha;
    grad_alpha_pos_class(~pos_mask) = -inf; 
    [max_val_pos, i_pos] = max(grad_alpha_pos_class);
    if isinf(max_val_pos), i_pos = 0; end

    grad_alpha_neg_class = grad_alpha;
    grad_alpha_neg_class(~neg_mask) = -inf;
    [max_val_neg, i_neg] = max(grad_alpha_neg_class);
    if isinf(max_val_neg), i_neg = 0; end
    
    u_best      = [];
    lambda_best = 0;
    delta_best  = 0;

    % Local refinement function for same-class pairs
    function scan_class(i_star, current_class_mask)
        if i_star == 0 || ~any(current_class_mask) || numel(find(current_class_mask)) < 2
            return;
        end

        for j = find(current_class_mask)' 
            if j == i_star || alpha(j) <= 1e-7
                continue; 
            end
            
            % Directional gradient
            grad_dot = grad_alpha(i_star) - grad_alpha(j);
            if grad_dot <= tol, continue; end

            % Combined decision and privileged space curvature
            denom = (K(i_star,i_star)+K(j,j)-2*K(i_star,j)) + ...
                    (1/gamma)*(Kstar(i_star,i_star)+Kstar(j,j)-2*Kstar(i_star,j));
            if denom <= 1e-12, continue; end

            lambda_raw  = grad_dot/denom;
            lambda_clip = min(lambda_raw, alpha(j)); % alpha_j decreases
            lambda_clip = max(0, lambda_clip);

            if lambda_clip < kappa, continue; end

            delta_D = lambda_clip*grad_dot - 0.5*lambda_clip^2*denom;
            
            if delta_D > delta_best 
                delta_best  = delta_D;
                lambda_best = lambda_clip;
                u_best      = zeros(2*N,1);
                u_best(i_star) =  1;
                u_best(j)      = -1;
            end
        end
    end
    
    scan_class(i_pos, pos_mask);
    scan_class(i_neg, neg_mask); 
    
    if verbosity && ~isempty(u_best)
        fprintf('[I2-fast] lambda=%.3e deltaD=%.3e\n', lambda_best, delta_best);
    end
end

% ======================= Type I3 Direction =======================
function [u_best, lambda_best, delta_best] = direccion_I3_fast(alpha, beta, y, grad_alpha, grad_beta, K, Kstar, gamma, tol, kappa, verbosity)
    % direccion_I3_fast - O(n) version
    % 1. Pre-selection: Choose pair (i*, j*) with opposite classes maximizing g_i + g_j.
    % 2. Refinement: Search for index k (beta) maximizing delta D.

    N = length(alpha);
    u_best      = [];
    lambda_best = 0;
    delta_best  = 0;

    % STEP 1: Select i* and j* (O(n) pass)
    g_pos = grad_alpha;
    g_neg = grad_alpha;
    g_pos(y==-1) = -inf;
    g_neg(y== 1) = -inf;

    [~, i_pos] = max(g_pos);
    [~, j_neg] = max(g_neg);

    if isinf(g_pos(i_pos)) || isinf(g_neg(j_neg)), return; end

    % STEP 2: Scan for beta index k (O(n) pass)
    i = i_pos; j = j_neg;

    % Direction sign depends on the class labels
    signAlpha = 1; % y(i)=+1, y(j)=-1 => +1, +1, -2 update
    if y(i)==-1 && y(j)==1
        signAlpha = -1; % Symmetric case
    end

    for k = 1:N
        % If beta_k decreases, it must be > 0.
        if signAlpha == 1 && beta(k) <= 0
            continue;
        end

        % Directional vector u (at most 3 non-zero indices)
        u_i =  signAlpha;
        u_j =  signAlpha;
        u_b = -2*signAlpha;

        % Directional gradient g^T u
        grad_dot = grad_alpha(i) + grad_alpha(j) - 2*grad_beta(k)*signAlpha;
        if grad_dot <= tol, continue; end

        % Denominator: u^T H u
        denom =  (K(i,i)+K(j,j)-2*K(i,j)) ...
               + (1/gamma)*( Kstar(i,i)+Kstar(j,j)+4*Kstar(k,k)+2*Kstar(i,j) ...
                             -4*Kstar(i,k)-4*Kstar(j,k) );
        if denom <= 1e-12, continue; end

        lambda_raw = grad_dot/denom;
        if signAlpha == 1 % beta_k decreases
            lambda_clip = min(lambda_raw, beta(k)/2);
        else              % beta_k increases, alpha_i/j decrease
            lambda_clip = min([lambda_raw, alpha(i), alpha(j)]);
        end
        
        if lambda_clip < kappa, continue; end

        delta_D = lambda_clip*grad_dot - 0.5*lambda_clip^2*denom;

        if delta_D > delta_best
            delta_best  = delta_D;
            lambda_best = lambda_clip;
            u_best = zeros(2*N,1);
            u_best(i)     = u_i;
            u_best(j)     = u_j;
            u_best(N+k)   = u_b;
        end
    end

    if verbosity && ~isempty(u_best)
        fprintf('[I3-fast] i=%d j=%d lambda=%.3e deltaD=%.3e\n', i, j, lambda_best, delta_best);
    end
end