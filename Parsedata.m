addpath('./amg-master/')
addpath('./Dataset/')
addpath('./DBSCAN Clustering/')

if ~exist('./data', 'dir')
    mkdir './data'
end


%%bitcoin
% clear all
% a = csvread('soc-sign-bitcoinotc.csv');
% a = a(:,1:3);
% a(:,3)=1;
% n = max(max(a))+ 1;
% A = zeros(n,n);
% for i = 1: size(a,1)
%     A(a(i,1)+1,a(i,2)+1)=1;
%     A(a(i,2)+1,a(i,1)+1)=1;
% end
% A = remove_zero_row(A,n);
% [P R W c] = AMG(sparse(A), 0.2, 5);
% 
% epsilon=0.5;
% MinPts=10;
% IDX=DBSCAN(full(A),epsilon,MinPts);
% edges = supernode(A,c);
% save('./data/coarsegraph_bitcoin')


% 
% % second p2p network
% clear all
% a = importdata('p2p-Gnutella08.txt');
% n = max(max(a))+ 1;
% A = zeros(n,n);
% for i = 1: size(a,1)
%     A(a(i,1)+1,a(i,2)+1)=1;
% end
% A = remove_zero_row(A,n);
% [P R W c] = AMG(sparse(A), 0.2, 5);
% 
% epsilon=0.5;
% MinPts=10;
% IDX=DBSCAN(full(A),epsilon,MinPts);
% edges = supernode(A,c);
% save('./data/coarsegraph_Gnutella08')
% 
% 
% % CA network
% clear all
% a = importdata('CA-GrQc.txt');
% n = max(max(a))+ 1;
% A = zeros(n,n);
% for i = 1: size(a,1)
%     A(a(i,1)+1,a(i,2)+1)=1;
% end
% A = remove_zero_row(A,n);
% [P R W c] = AMG(sparse(A), 0.2, 5);
% 
% epsilon=0.5;
% MinPts=10;
% IDX=DBSCAN(full(A),epsilon,MinPts);
% edges = supernode(A,c);
% save('./data/coarsegraph_CA')
% 
% % 
% %facebook
% clear all
% a = importdata('facebook_combined.txt');
% n = max(max(a)) + 1;
% A = zeros(n,n);
% for i = 1: size(a,1)
%     A(a(i,1)+1,a(i,2)+1)=1;
% end
% A = remove_zero_row(A,n);
% [P R W c] = AMG(sparse(A), 0.2, 5);
% 
% epsilon=0.5;
% MinPts=10;
% IDX=DBSCAN(full(A),epsilon,MinPts);
% edges = supernode(A,c);
% save('./data/coarsegraph_facebook')
% 
% 
% %p2p
% clear all
% a = importdata('p2p-Gnutella04.txt');
% n = max(max(a)) + 1;
% A = zeros(n,n);
% % A(a(:,1)+1,a(:,2)+1) = 1;
% for i = 1:size(a,1)
%     A(a(i,1)+1,a(i,2)+1)=1;
% end
% A=(A +A')/2;
% A = remove_zero_row(A,n);
% [P R W c] = AMG(sparse(A), 0.2, 5);
% 
% epsilon=0.5;
% MinPts=10;
% IDX=DBSCAN(full(A),epsilon,MinPts);
% edges = supernode(A,c);
% save('./data/coarsegraph_p2p')
% 
% %wiki
% clear all
% a = importdata('Wiki-Vote.txt');
% n = max(max(a)) + 1;
% A = zeros(n,n);
% for i = 1: size(a,1)
%     A(a(i,1)+1,a(i,2)+1)=1;
% end
% A = remove_zero_row(A,n);
% [P R W c] = AMG(sparse(A), 0.2, 5);
% 
% epsilon=0.5;
% MinPts=10;
% IDX=DBSCAN(full(A),epsilon,MinPts);
% edges = supernode(A,c);
% save('./data/coarsegraph_wiki')
% 

%%Email
clear all
a = importdata('email-Eu-core.txt');
n = max(max(a)) + 1;
A = zeros(n,n);
for i = 1: size(a,1)
    A(a(i,1)+1,a(i,2)+1)=1;
end
A = remove_zero_row(A,n);
[P R W c] = AMG(sparse(A), 0.2, 5);
epsilon=0.5;
MinPts=4;
IDX=DBSCAN(full(A),epsilon,MinPts);
edges = supernode(A,c);
save('./data/email-Eu-core');
exit(0);

function edges = supernode(A,c)
    edges = zeros(1,5);
    for i = 1:5
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
        edges(1,i) = sum(sum(F));
    end
end

function A = remove_zero_row(A,n)
    % check if any columns are zero columns
    B = sum(A)==0;
    indices = [];
    for i = 1:n
        if ~B(i)
            indices = [indices, i];
        end 
    end
    A = A(indices,indices);
end
