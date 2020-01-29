function sim = GS_RW_Plain(A,B,c,flag,p,q)
% % % % using random walk kernel to compute the similarity between A & B
n1 = size(A,1);
n2 = size(B,1);

 [u1,lam1] = eigs(A,1,'lm');
 [u2,lam2] = eigs(B,1,'lm');
 c = 1/(abs(lam1 * lam2)+1);

if nargin<4
    flag = [1 1 1 1];%directly inverse;linear system;sylvester;eigens
end
if nargin<6
    q = {ones(n1,1)/n1,ones(n2,1)/n2};
end
if nargin<5
    p = {ones(n1,1)/n1,ones(n2,1)/n2};
end
if nargin<3
    [u1,lam1] = eigs(A,1,'lm');
    [u2,lam2] = eigs(B,1,'lm');
    c = 1/(abs(lam1 * lam2)+1);
end
p1 = p{1};
p2 = p{2};
q1 = q{1};
q2 = q{2};
   
if flag(1)==1%direct inverse
    X = kron(A,B);
    qx = kron(q1,q2);
    px = kron(p1,p2);
    sim(1,1) = qx' * inv(eye(n1 * n2) - c * X) * px;
end
if flag(2)==1%linear system
    X = kron(A,B);
    qx = kron(q1,q2);
    px = kron(p1,p2);
    x = rand(n1*n2,1);
    for i=1:200
        x = px + c * (X * x);
    end
    sim(1,2) = qx' * x;
end

if flag(4)==1%eigen-decomposition, for symmetric matrix only
    r1 = rank(A);
    r2 = rank(B);
    [U1,Lam1] = eigs(A,r1,'lm');
    [U2,Lam2] = eigs(B,r2,'lm');
    
    x1 = q1' * U1;
    x2 = q2' * U2;
    y1 = U1' * p1;
    y2 = U2' * p2;
    
    l = 1./(1-c*kron(diag(Lam1),diag(Lam2)));
    x = kron(x1,x2);
    y = kron(y1,y2);
    sim(1,4) = sum(x'.*l.*y);
%     L1 = diag(kron(Lam1,Lam2));
%     
%     H2 = diag(1./(1-c*L1));
%     sim0 = kron(x1,x2) * H2 * kron(y1,y2);
%     disp(sim0-sim(1,4))
end    
