clc; clear; close all;

% ------------------------------------------------------
% --- Datos de ejemplo de MNIST (train_reducted) -----
rng(42);
S = load('train_reducted.mat');  % debe contener la estructura 'train'
train = S.train;
X  = double(train.X_train);      % N x 100  (10x10 vectorizado)
lbl = train.y_train(:);          % N x 1    (etiquetas en {-1,+1})
PI = double(train.PI_train);     % N x 5    (información privilegiada)

% Lista de tamaños de muestra a probar
sample_sizes = [1000, 2000, 4000, 7000, 12000, 22000, 35000, 50000];

% Inicializar resultados
results = table('Size',[numel(sample_sizes) 3], ...
    'VariableTypes',{'double','double','double'}, ...
    'VariableNames',{'NumSamples','Time_aSMO','Obj_aSMO'});

% --- Parámetros del modelo ---
C = 10;
gamma = 1;
sgmPlus = 1;
sgmStar = 0.5;

% Opciones para el optimizador aSMO
opts_asmo = struct();
opts_asmo.tol = 1e-4;       
opts_asmo.maxIter = 50000;   
opts_asmo.verbosity = 0;    
opts_asmo.kappa = 1e-7;     

% Bucle principal sobre tamaños
for k = 1:numel(sample_sizes)
    Ntarget = sample_sizes(k);

    % --- Muestreo estratificado con reemplazo si hace falta ---
    idx_pos = find(lbl==+1);
    idx_neg = find(lbl==-1);
    assert(~isempty(idx_pos) && ~isempty(idx_neg), 'Faltan muestras de alguna clase.');

    m_pos = ceil(Ntarget/2);
    m_neg = Ntarget - m_pos;

    % Con reemplazo (true) para permitir duplicados cuando se supera el total disponible
    sel_pos = randsample(idx_pos, m_pos, true);
    sel_neg = randsample(idx_neg, m_neg, true);
    sel = [sel_pos; sel_neg];
    sel = sel(randperm(numel(sel)));   % barajar mezcla final

    fv     = normalize(X(sel,:));     % características de decisión (N x 100)
    fvStar = normalize(PI(sel,:));    % info privilegiada (N x 5)
    y      = lbl(sel);
    Nused  = numel(sel);

    fprintf('\n===== N = %d muestras (requested %d) =====\n', Nused, Ntarget);

    % Kernels
    K     = exp(-pdist2(fv, fv).^2 / (2 * sgmPlus^2));
    % (Kstar no se usa para fval_smo aquí; lo calcularías si lo necesitas)
    % Kstar = exp(-pdist2(fvStar, fvStar).^2 / (2 * sgmStar^2));

    % --- Entrenar con SMO+ ---
    fprintf('Entrenando con SMO+...\n');
    t0 = tic;
    [z_smo, fval_smo, b] = solve_asmo(fv, fvStar, y, C, gamma, sgmPlus, sgmStar, opts_asmo);
    time_smo = toc(t0);
    fprintf('Valor función objetivo (SMO+): %.6f (%.3f s)\n', fval_smo, time_smo);

    % Guardar resultados (asignación por columnas)
    results.NumSamples(k) = Nused;
    results.Time_aSMO(k)  = time_smo;
    results.Obj_aSMO(k)   = fval_smo;
end

disp('=== Resultados finales ===');
disp(results);

% Guardar tabla en CSV
writetable(results, 'results_asmo_scaling.csv');


%%
% ----------------- Plots -----------------
% 1) Training time vs number of samples
figure('Name','Training time vs Number of samples');
plot(results.NumSamples, results.Time_aSMO, '-o', 'LineWidth', 1.5);
grid on; 
xlabel('Number of samples'); 
ylabel('Training time (s)');
title('Training time comparison: aSMO large scale');
legend('aSMO','Location','northwest');

