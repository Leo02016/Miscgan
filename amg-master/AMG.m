function [P R W c] = AMG(fW, beta, NS)
%
% Graph coarsening using Algebraic MultiGrid (AMG)
%
% Usage: 
%   [P R W c] = AMG(fW, beta, NS)
%
% Inputs:
%   fW      - affinity matrix at the finest scale. 
%             Matrix must be symmetric, sparse and have non-negative
%             entries.
%   beta    - coarsening factor (typical value ~.2)
%   NS      - number of scales / levels
%
% Outputs
%   P       - interp matrices (from coarse to fine)
%   R       - restriction matrices (from fine to coarse)
%   W       - affective affinity matrix at each scale (W{1}=fW)
%   c       - coarse nodes selected at each scale
%
% Example:
%   Image super-pixels
%
%   img = im2double(imread('football.jpg'));
%   sz = size(img);
%   % forming 4-connected graph over image grid
%   [ii jj] = sparse_adj_matrix(sz(1:2), 1, 1, 1);
%   fimg = reshape(img,[],sz(3));
%   wij = fimg(ii,:)-fimg(jj,:);
%   wij = sum(wij.^2, 2);
%   wij = exp(-wij./(2*mean(wij)));
%   fW = sparse(ii, jj, wij, prod(sz(1:2)), prod(sz(1:2)));
%
%   % forming AMG
%   [P R W c] = AMG(fW, .2, 5);
%
%   % super pixels
%   [~, sp] = max( P{1}*(P{2}*(P{3}*(P{4}*P{5}))), [], 2);
%
%   figure;
%   subplot(121);imshow(img);title('input image');
%   subplot(122);imagesc(reshape(sp,sz(1:2)));
%   axis image;colormap(rand(numel(c{5}),3));
%   title('super pixels');
%
%
%  Copyright (c) Bagon Shai
%  Many thanks to Meirav Galun.
%  Department of Computer Science and Applied Mathmatics
%  Wiezmann Institute of Science
%  http://www.wisdom.weizmann.ac.il/
% 
%  Permission is hereby granted, free of charge, to any person obtaining a copy
%  of this software and associated documentation files (the "Software"), to deal
%  in the Software without restriction, subject to the following conditions:
% 
%  1. The above copyright notice and this permission notice shall be included in
%      all copies or substantial portions of the Software.
%  2. No commercial use will be done with this software.
%  3. If used in an academic framework - a proper citation must be included.
%
% @ELECTRONIC{bagon2012,
%  author = {Shai Bagon},
%  title = {Matlab implementation for AMG graph coarsening},
%  url = {http://www.wisdom.weizmann.ac.il/~bagon/matlab.html#AMG},
%  owner = {bagon},
%  version = {}, <-please insert version number from VERSION.txt
% }
%
% 
%  The Software is provided "as is", without warranty of any kind.
% 
%  May 2011
% 


n = size(fW,1);

P = cell(1,NS);
c = cell(1,NS);
W = cell(1,NS);

% make sure diagonal is zero
W{1} = fW - spdiags( spdiags(fW,0), 0, n, n);


fine = 1:n;

for si=1:NS
    
    [tmp_c P{si}] = fine2coarse(W{si}, beta);    
    c{si} = fine(tmp_c);
    
    if si<NS
        
        W{si+1} = P{si}'*W{si}*P{si};
        
        W{si+1} = W{si+1} - spdiags( spdiags(W{si+1},0), 0, size(W{si+1},1), size(W{si+1},2));
        fine = c{si};
    end
end
% restriction matrices
R = cellfun(@(x) spmtimesd(x', 1./sum(x,1), []), P, 'UniformOutput', false);


%-------------------------------------------------------------------------%
function [c P] = fine2coarse(W, beta)
%
% Performs one step of coarsening
%

n = size(W,1);
% weight normalization
nW = spmtimesd(W, 1./full(sum(W,2)), []);

% % % select coarse nodes (mex implementation)
c = ChooseCoarseGreedy_mex(nW, randperm(n), beta);

% % c = false(1,n);
% % sum_jc = zeros(n,1);
% % % for ii=1:n % lexicographic order - any better ideas?
% % for ii = randperm(n) % random order    
% %     if sum_jc(ii) <= beta
% %         c(ii)=true;
% %         sum_jc = sum_jc + nW(:,ii); 
% %     end
% % end


% compute the interp matrix
ci=find(c);
P = W(:,ci);
P = spmtimesd(P, 1./full(sum(P,2)), []);
% make sure coarse points are directly connected to their fine counterparts
[jj ii pji] = find(P'); 
sel = ~c(ii); % select only fine points
mycat = @(x,y) vertcat(x(:),y(:));
P = sparse( mycat(jj(sel), 1:sum(c)),...
    mycat(ii(sel), ci),...
    mycat(pji(sel), ones(1,sum(c))), size(P,2), size(P,1))';
