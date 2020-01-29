#include "mex.h"
#include <string.h> // for memset 

/*
 * Choose variables for coarse grid 
 * Greedy algorithm
 *
 * After selection, 
 * 
 * \sum_{j\in coarse} c_ij > beta \sum_j c_ij for all i
 *
 * Starting with an empty coarse set, we sequentially add variables from fine
 * That their \sum_{j\in coarse} c_ij (to current estimate of coarse) 
 * is smaller than: beta \sum_ij c_ij
 *
 * Usage
 *      c = ChooseCoarseGreedy_mex(nC, ord, beta)
 *
 * Inputs
 *      nC     - sparse, n x n matrix, with nC_ij = c_ij / (sum_j c_ij)
 *      ord    - n elements vector of indices, 
 *               indicates the order at which the fine variables should be 
 *               considered in the greedy algorithm.
 *      beta   - scalar 0 < beta < 1 governing the coarsening rate (usually 0.2)
 *
 * Output
 *      c      - n element indicator vector for the chosen coarse variables
 *
 * To compile:
 * >> mex -largeArrayDims -O ChooseCoarseGreedy_mex.cpp
 *
 */


#line   __LINE__  "AdjustPdeg"

#define     STR(s)      #s  
#define     ERR_CODE(a,b)   a ":" "line_" STR(b)


// INPUTS
enum {
    iC = 0,
    iO,
    iB,
    nI
};

// OUTPUTS
enum {
    oC = 0,
    nO
};     


void
mexFunction(
    int nout,
    mxArray* pout[],
    int nin,
    const mxArray* pin[])
{
    if ( nin != nI )
         mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"must have %d inputs", nI);
    if (nout==0)
        return;
    if (nout != nO )
         mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"must have exactly %d output", nO);
    
    if ( mxIsComplex(pin[iC]) || !mxIsSparse(pin[iC]) || 
            ! mxIsDouble(pin[iC]) || mxGetNumberOfDimensions(pin[iC]) != 2 )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"nC must be a sparse double matrix");
    
    mwSize n = mxGetM(pin[iC]);
    
    if ( mxGetN(pin[iC]) != n )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"nC must be a square matrix");
    
    if ( mxIsComplex(pin[iO]) || mxIsSparse(pin[iO]) || 
            ! mxIsDouble(pin[iO]) || mxGetNumberOfElements(pin[iO]) != n )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"ord must be a double vector with %d elements", n);
    
    if ( mxGetNumberOfElements(pin[iB]) != 1 )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"beta must be a scalar");
    
    double beta = mxGetScalar(pin[iB]);
    if ( beta <= 0 || beta >= 1 )
        mexErrMsgIdAndTxt(ERR_CODE(__FILE__, __LINE__),"beta must be in the range (0,1)");
    
    // allocate space for sum_jc
    double* sum_jc = new double[n];
    memset(sum_jc, 0, n*sizeof(double));
    
 
    // allocate space for indicator vector c
    pout[oC] = mxCreateLogicalMatrix(1, n);
    mxLogical* c =  mxGetLogicals(pout[oC]);
    
    double* pr = mxGetPr(pin[iC]);
    mwIndex* pir = mxGetIr(pin[iC]);
    mwIndex* pjc = mxGetJc(pin[iC]);
    
    double* ord = mxGetPr(pin[iO]);
    
    for ( mwIndex ii(0); ii < n ; ii++ ) {
        mwIndex current = static_cast<mwIndex>(ord[ii])-1; // convert from matlab's 1-based indexing to 0-based indexing
        mxAssert( current >= 0 && current < n, "order contains out of range indices");
        
        if ( sum_jc[current] <= beta ) {
            // we need to add current to coarse
            c[current] = true;
            
            // update sum_jc accordingly                          
            for ( mwIndex jj = pjc[current] ; jj < pjc[current+1] ; jj++ ) {
                mwIndex row = pir[jj];
                sum_jc[row] += pr[jj];
            }
        }
    }    
    
    
    delete[] sum_jc;    
}
 
