function [X,err,gradnorm,time] = adaptive_iht(A,m,n,y,s,k,iterations,epsilon,X0,Xstar,displ)

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

function g = grad(Z)
    g = reshape(A'*(A*Z(:) - y),[m,n]);
end

function T = retraction(Z)
    [~,inds] = sort(rownorms(Z),'desc');
    T = zeros(m,n);
    [uu,ss,vv] = svd(Z(inds(1:s),:),'econ');
    T(inds(1:s),:) = uu(:,1:k)*ss(1:k,1:k)*vv(:,1:k)';
end

X = X0;
err = norm(X - Xstar,'fro')/norm(Xstar,'fro');
newerr = err;
g = grad(X);
gnorm = norm(g,'fro');
gradnorm = gnorm;
tic;
time = toc;
oldnorm = 0;
it = 1;
fprintf('\n Adaptive IHT, m = %i, n = %i, s = %i, k = %i. \n \n',m,n,s,k);
while it < iterations && gnorm > epsilon && gnorm < 1e+2 && abs(gnorm - oldnorm) > 0.0001*epsilon && newerr >= 1e-5
    
    oldnorm = gnorm;
    alpha = 10;
    c = 0.0001;
    R = retraction(X - alpha*g);
    while cost(X) - cost(R) < alpha*c*gnorm^2 && alpha > 1e-12
        alpha = alpha/2;
        R = retraction(X - alpha*g);
    end
    if alpha < 1e-12
        X = retraction(X - g);
    else
        X = R;
    end
    g = grad(X);
    newerr = norm(X - Xstar,'fro')/norm(Xstar,'fro');
    err = [err; newerr];
    gnorm = norm(g,'fro');
    gradnorm = [gradnorm; gnorm];
    time = [time; toc];
    if displ
        if mod(it,1) == 0
            fprintf('Step: %i,\t Relative Error: %d,\t Gradient: %d \n',it,err(it+1),gnorm);
        end
    end
    
    it = it + 1;
end

end
