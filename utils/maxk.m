function maxentries = maxk(v,k)

list = sort(v,'descend');
maxentries = list(1:k);

end