function [X,err,gradnorm,time] = riemannian_proximal_gradient(A,m,n,y,k,lambda,iterations,epsilon,X0,Xstar,displ)

M = size(A,1);
s = min(m,round((M + k^2 - k*n)/k));

if ~exist('iterations','var')
    iterations = 1000;
end

if ~exist('epsilon','var')
    epsilon = 1e-7;
end

if ~exist('X0','var') || isempty(X0)
    X0 = zeros(m,n);
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

function N = nnz(Z)
    N = 0;
    for i = 1:m
        if norm(Z(i,:)) ~= 0
            N = N + 1;
        end
    end
end

function T = retraction(Z)
    [uu,ss,vv] = svd(Z,'econ');
    T = uu(:,1:k)*ss(1:k,1:k)*vv(:,1:k)';
end

function T = softthresh(Z,thresh)
    T = zeros(m,n);
    for i = 1:m
        if norm(Z(i,:),'fro') > thresh*lambda
            T(i,:) = (1 - (thresh*lambda)/norm(Z(i,:)))*Z(i,:);
        end
    end
end

X = X0;
err = norm(X - Xstar,'fro')/norm(Xstar,'fro');
newerr = err;
g = rgrad(X);
gnorm = norm(g,'fro');
gradnorm = gnorm;
tic;
time = toc;
oldnorm = 0;
it = 1;
fprintf('\n Riemannian Proximal Gradient, m = %i, n = %i. \n \n',m,n);
while it < iterations && gnorm > epsilon && gnorm < 1e+2 && abs(gnorm - oldnorm) > 0.0001*epsilon && newerr >= 1e-5

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
    rws = maxk(rownorms(X),k);
    tau = (0.99^it)*rws(k);
    X = softthresh(X,tau);
    g = rgrad(X);
    newerr = norm(X - Xstar,'fro')/norm(Xstar,'fro');
    err = [err; newerr];
    gnorm = norm(g,'fro');
    gradnorm = [gradnorm; gnorm];
    time = [time; toc];
    if displ
        if mod(it,1) == 0
            fprintf('Step: %i,\t Relative Error: %d,\t Riemannian Gradient: %d, NNZ: %i \n',it,err(it+1),gnorm,nnz(X));
        end
    end
    
    it = it + 1;
end

end
