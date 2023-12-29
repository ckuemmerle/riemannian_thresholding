function [X,outs] = riemannian_adaptive_iht(A,m,n,y,opts)
% Implements Adaptive Riemannian Iterative Hard Thresholding for recovery
% of simultaneously low-rank and row-sparse matrices as studied in [1].
%
% Input:
%   m : number of rows of target matrix
%   n : number of columns of target matrix
%   A : measurement operator matrix (it is a M x (m* n) matrix)
%   y : measurement vector (M x 1 vector)
% opts: structure array with algorithmic options
%        - N0_firstorder: Max. number of iterations
%        - tol          : Tolerance parameter
%        - X0           : Initial matrix
%        - K1           : Row-sparsity estimate
%        - r            : Rank estimate
%        - verbose      : boolean, determines if printing output or not
%        - saveiterates : boolean, determines if all iterates are saved and
%        returned 
% Output:
%   X : Recovered/Estimated matrix (if saveiterates == false), otherwise,
%       cell array with all estimated matrices.
% outs: Some algorithmic quantities
%
% Reference:
% [1] Henrik Eisenmann, Felix Krahmer, Max Pfeffer, and André Uschmajew. 
% Riemannian thresholding methods for row-sparse and low-rank matrix recovery.
% Numerical Algorithms, pages 1–25, 2022,
% https://doi.org/10.1007/s11075-022-01433-5.

s       = opts.K1;    % Target (Maximal) Row sparsity of X
k       = opts.r;     % Target (Maximal) rank of X

if isfield(opts,'N0')  && ~isempty(opts.N0_firstorder)
    N0 = opts.N0_firstorder;
else
    N0 = 1000;
end

if isfield(opts,'tol')  && ~isempty(opts.tol)
    tol = opts.tol;
else
    tol = 1e-7;
end

if isfield(opts,'X0')  && ~isempty(opts.X0)
    X0 = opts.X0;
else
    X0 = zeros(m,n);
end
if isfield(opts,'saveiterates')  && ~isempty(opts.saveiterates) 
    saveiterates = opts.saveiterates;
else
    saveiterates = false;
end
if isfield(opts,'verbose')  && ~isempty(opts.verbose) 
    verbose = opts.verbose;
else
    verbose = true;
end

function f = cost(Z)
    f = .5*dot(A*Z(:) - y,A*Z(:) - y);
end

function g = rgrad(Z)
    [uu,ss,vv] = svd(Z,'econ');
    uu = uu(:,1:k);
    ss = ss(1:k,1:k);
    vv = vv(:,1:k);
    g = reshape(A'*(A*Z(:) - y),[m,n]);
    if ss == 0
        [~,inds] = sort(rownorms(g),'desc');
        T = zeros(m,n);
        T(inds(1:s),:) = g(inds(1:s),:);
        [uuu,sss,vvv] = svd(T,'econ');
        g = uuu(:,1:k)*sss(1:k,1:k)*vvv(:,1:k)';
    else
        g = (eye(m) - uu*uu')*g*(vv*vv') + (uu*uu')*g;
    end
end

function T = retraction(Z)
    [~,inds] = sort(rownorms(Z),'desc');
    T = zeros(m,n);
    [uu,ss,vv] = svd(Z(inds(1:s),:),'econ');
    T(inds(1:s),:) = uu(:,1:k)*ss(1:k,1:k)*vv(:,1:k)';
end

X = X0;
%err = norm(X - Xstar,'fro')/norm(Xstar,'fro');
%newerr = err;
Xcell = cell(1,N0);
g = rgrad(X);
gnorm = norm(g,'fro');
gradnorm = gnorm;
tic;
time = [];
oldnorm = 0;
it = 1;
fprintf('\n Riemannian Adaptive IHT, m = %i, n = %i, s = %i, k = %i. \n \n',m,n,s,k);
while it <= N0

    oldnorm = gnorm;
    alpha = 10;
    c = 0.0001;
    R = retraction(X - alpha*rgrad(X));
    while cost(X) - cost(R) < alpha*c*norm(rgrad(X),'fro')^2 && alpha > 1e-12
        alpha = alpha/2;
        R = retraction(X - alpha*rgrad(X));
    end
    if alpha < 1e-12
        X = retraction(X - rgrad(X));
    else
        X = R;
    end
    g = rgrad(X);
    gnorm = norm(g,'fro');
    gradnorm = [gradnorm; gnorm];
    time = [time; toc];
    if saveiterates
        Xcell{it} = X;
    end
    if gnorm <= tol || gnorm >= 1e+2 || abs(gnorm - oldnorm) <= 0.0001*tol
        N0 = it;
    end
    if verbose > 1 || (verbose == 1 && (it ==1 || it == N0))
        %if mod(it,1) == 0
            fprintf('Step: %i,\t Riemannian Gradient: %d \n',it,gnorm);
        %end
    end
    it = it + 1;
end
outs = struct;
if saveiterates
    outs.X               = Xcell(1:N0);
else
    outs.X = X;
end
outs.N = N0;
outs.gradnorm = gradnorm;
outs.time = time;
end
