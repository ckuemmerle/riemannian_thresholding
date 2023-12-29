function rn = rownorms(A)

m = size(A,1);
rn = zeros(m,1);

for i = 1:m
    rn(i) = norm(A(i,:));
end

end