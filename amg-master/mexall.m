function mexall()
%
% mex cpp associated code
%

mex -O -largeArrayDims ChooseCoarseGreedy_mex.cpp
mex -O -largeArrayDims spmtimesd.cpp
exit(0);
