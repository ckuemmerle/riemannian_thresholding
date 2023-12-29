function [X,err,gradnorm,time] = adaptive_iht(Mat,A,B,m,n,y,s,k,iterations,epsilon,X0,Xstar,displ)

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
fprintf('\n Adaptive IHT, m = %i, n = %i, s = %i, k = %i. \n \n',m,n,s,k);
while it < iterations && gnorm > epsilon && gnorm < 1e+2 && abs(gnorm - oldnorm) > 0.0001*epsilon && newerr >= 1e-5
    
    oldnorm = gnorm;
    alpha = 10;
    c = 0.0001;
    [UR,SR,VR] = retraction(U*S*V' - alpha*g);
    while cost(U,S,V) - cost(UR,SR,VR) < alpha*c*gnorm^2 && alpha > 1e-12
        alpha = alpha/2;
        [UR,SR,VR] = retraction(U*S*V' - alpha*g);
    end
    if alpha < 1e-12
        [U,S,V] = retraction(U*S*V' - g);
    else
        U = UR;
        S = SR;
        V = VR;
    end
    g = grad(U,S,V);
    newerr = norm(U*S*V' - Xstar,'fro')/norm(Xstar,'fro');
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

X = U*S*V';

end
