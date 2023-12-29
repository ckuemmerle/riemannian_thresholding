function [X,err,gradnorm,time] = riemannian_proximal_gradient(A,B,m,n,y,k,lambda,iterations,epsilon,X0,Xstar,displ)

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

function f = cost(U,S,V)
    sigma = diag(S);
    f = .5*dot(vec((A*U).*(B*V)*sigma) - y,vec((A*U).*(B*V)*sigma) - y);
end

function [U,S,V,Uperp,Vperp,M,R1,R2,g] = rgrad(U,S,V)
    if S(1,1) == 0
        Mat = zeros(size(A,1),m*n);
        for i = 1:size(A,1)
            K = kron(B(i,:),A(i,:));
            Mat(i,:) = K(:);
        end
        X = reshape(-Mat'*y,[m,n]);
        [~,inds] = sort(rownorms(X),'desc');
        T = zeros(m,n);
        T(inds(1:s),:) = X(inds(1:s),:);
        [U,M,V] = svd(T,'econ');
        U = U(:,1:k);
        M = M(1:k,1:k);
        V = V(:,1:k);
        Uperp = zeros(m,k);
        Vperp = zeros(n,k);
        R1 = zeros(k,k);
        R2 = zeros(k,k);
    else
        sigma = diag(S);
        z = vec((A*U).*(B*V)*sigma) - y;
        zUab = zeros(k,n);
        zabV = zeros(m,k);
        zUabV = zeros(k,k);
        for i = 1:size(A,1)
            Ua = U'*A(i,:)';
            bV = B(i,:)*V;
            zUab = zUab + z(i)*Ua*B(i,:);
            zabV = zabV + z(i)*A(i,:)'*bV;
            zUabV = zUabV + z(i)*Ua*bV;
        end
        [Uperp,R1] = qr(zabV - U*zUabV,0);
        Uperp = Uperp(:,1:k);
        R1 = R1(1:k,:);
        [Vperp,R2] = qr(zUab' - V*zUabV',0);
        Vperp = Vperp(:,1:k);
        R2 = R2(1:k,:);
        M = zUabV;
    end
    g = [U, Uperp]*[M, R2'; R1, zeros(k,k)]*[V'; Vperp'];
end

function N = nnz(Z)
    N = 0;
    for i = 1:m
        if norm(Z(i,:)) ~= 0
            N = N + 1;
        end
    end
end

function [UU,SS,VV] = retraction(U,S,V,Uperp,Vperp,M,R1,R2)
    [uu,ss,vv] = svd([S-M, -R2'; -R1, zeros(k,k)],'econ');
    UU = [U, Uperp]*uu(:,1:k);
    SS = ss(1:k,1:k);
    VV = [V, Vperp]*vv(:,1:k);
end

function [UU,SS,VV] = softthresh(U,S,V,thresh)
    Q = U*S;
    T = zeros(m,k);
    for i = 1:m
        if norm(Q(i,:),'fro') > thresh*lambda
            T(i,:) = (1 - (thresh*lambda)/norm(Q(i,:)))*Q(i,:);
        end
    end
    [UU,SS,VV] = svd(T,'econ');
    UU = UU(:,1:k);
    SS = SS(1:k,1:k);
    VV = V*VV(:,1:k);
end

[U,S,V] = svd(X0,'econ');
U = U(:,1:k);
S = S(1:k,1:k);
V = V(:,1:k);
err = norm(U*S*V' - Xstar,'fro')/norm(Xstar,'fro');
newerr = err;
[U,S,V,Uperp,Vperp,M,R1,R2,g] = rgrad(U,S,V);
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
    [RU,RS,RV] = retraction(U,S,V,Uperp,Vperp,alpha*M,alpha*R1,alpha*R2);
    while cost(U,S,V) - cost(RU,RS,RV) < alpha*c*gnorm^2 && alpha > 1e-12
        alpha = alpha/2;
        [RU,RS,RV] = retraction(U,S,V,Uperp,Vperp,alpha*M,alpha*R1,alpha*R2);
    end
    if alpha < 1e-12
        [U,S,V] = retraction(U,S,V,Uperp,Vperp,M,R1,R2);
    else
        U = RU;
        S = RS;
        V = RV;
    end
    rws = maxk(rownorms(U*S),k);
    tau = (0.999^it)*rws(k);
    [U,S,V] = softthresh(U,S,V,tau);
    [U,S,V,Uperp,Vperp,M,R1,R2,g] = rgrad(U,S,V);
    newerr = norm(U*S*V' - Xstar,'fro')/norm(Xstar,'fro');
    err = [err; newerr];
    gnorm = norm(g,'fro');
    gradnorm = [gradnorm; gnorm];
    time = [time; toc];
    if displ
        if mod(it,1) == 0
            fprintf('Step: %i,\t Relative Error: %d,\t Riemannian Gradient: %d, NNZ: %i \n',it,err(it+1),gnorm,nnz(U));
        end
    end
        
    it = it + 1;
end

end
