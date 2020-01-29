% B = randi([0 1], 10,10);
% A = (B + B');
% A=A~=0;
A = sprandn(10,10,0.2);
A = A + A';
A = A~=0;
A = full(A);
c1 = [2,4,5,7,9,10];
c2 = [1 , 3 ,4, 6];
c3 = [1, 4];
c = {c1,c2,c3};
supernode = cell(1,3);
for i = 1:3
    F = A;
    n = size(F,1);
    m = size(c{1,i},2);
    E = zeros(m,n);
    prev = 1;
    for j = 1: m
        later = c{1,i}(j);
        for k = prev:later
           E(j,:) = E(j,:) + F(k,:);
        end
        prev = later + 1;
    end
    F = zeros(m,m);
    prev = 1;
    for j = 1: m
        later = c{1,i}(j);
        for k = prev:later
           F(:,j) = F(:,j) + E(:,k);
        end
        prev = later + 1;
    end
    F = F~=0;
    supernode{1,i} = F;
end
disp('Finish');