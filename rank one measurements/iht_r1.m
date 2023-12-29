function [X,err,gradnorm,time] = iht(Mat,A,B,m,n,y,s,k,iterations,epsilon,X0,Xstar,displ)

M = size(A,1);

if ~exist('iterations','var')
    iterations = 1000;
end

if ~exist('epsilon','var')
    epsilon = 1e-7;
end

if ~exist('X0','var') || isempty(X0)
    X0 = zeros(m,n);
end

function f = cost(U,S,V)
    z = zeros(M,1);
    for i = 1:M
        z(i) = (A(i,:)*U)*S*(V'*B(i,:)');
    end
    f = .5*dot(z - y,z - y);
end

function g = grad(U,S,V)
    z = zeros(M,1);
    for i = 1:M
        z(i) = (A(i,:)*U)*S*(V'*B(i,:)');
    end
    g = reshape(Mat'*(z - y),[m,n]);
end

function [UU,SS,VV] = retraction(Z) 
    [~,inds] = sort(rownorms(Z),'desc');
    UU = zeros(m,k);
    [uu,ss,vv] = svd(Z(inds(1:s),:),'econ');
    UU(inds(1:s),:) = uu(:,1:k);
    SS = ss(1:k,1:k);
    VV = vv(:,1:k);
end

[U,S,V] = svd(X0,'econ');
err = norm(X0 - Xstar,'fro')/norm(Xstar,'fro');
newerr = err;
U = U(:,1:k);
S = S(1:k,1:k);
V = V(:,1:k);
g = grad(U,S,V);
gnorm = norm(g,'fro');
gradnorm = gnorm;
tic;
time = toc;
oldnorm = 0;
it = 1;
fprintf('\n IHT, m = %i, n = %i, s = %i, k = %i. \n \n',m,n,s,k);
while it < iterations && gnorm > epsilon && gnorm < 1e+2 && abs(gnorm - oldnorm) > 0.0001*epsilon && newerr >= 1e-5
    
    oldnorm = gnorm;
    [U,S,V] = retraction(U*S*V' - g);
    g = grad(U,S,V);
    gnorm = norm(g,'fro');
    newerr = norm(U*S*V' - Xstar,'fro')/norm(Xstar,'fro');
    err = [err; newerr];
    gradnorm = [gradnorm; gnorm];
    time = [time; toc];
    if displ
        if mod(it,1) == 0
            fprintf('Step: %i,\t Relative Error: %d,\t Gradient: %d \n',it,err(it+1),gnorm);
        end
    end
    
    it = it + 1;
end

X = U*S*V';

end