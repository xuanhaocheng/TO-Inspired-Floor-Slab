// 
// revision 12:13 1.15.2024
// 

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include "mex.h"
#include <assert.h>
#include "matrix.h"
#include <cuda_runtime.h>


int estWorkSpace(int nelx0, int nely0, int nelz0);

__global__ void Assembly_F0_device_1( double *F0_d1, double *pasS_d1,int *cVec_d1, double g,int pasS_length0, int nelx,int nely,int nelz);
__global__ void Assembly_F0_device_2( double *F0_d1, double *pasS_d1,int *cVec_d1, double g,int pasS_length0, int nelx,int nely,int nelz);
__global__ void Assembly_F0_device_3( double *F0_d1, double *pasS_d1,int *cVec_d1, double g,int pasS_length0, int nelx,int nely,int nelz);
void Assembly_F0_1(double*F0,double *pasS,int pasS_length, mxInt32 *cVec0,double g, int nelx, int nely, int nelz, int direction);

FILE *flog;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
                            
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      flog=fopen("mgcg9.log", "a");
      fprintf(flog,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      fclose(flog);
      if (abort) exit(code);
   }
}

#define check() gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize())
/*-----------------------------------------------------------------------*/
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *F0, *pasS;
    mxInt32 *cVec;
    
    int nelx, nely, nelz,direction;
    double g;

    F0 = mxGetDoubles(prhs[0]);
    int mrows = mxGetM(prhs[0]);
    int ncols = mxGetN(prhs[0]);
    // mexPrintf("\n F0 %d %d input dimensions.\n", mrows, ncols);
    pasS = mxGetDoubles(prhs[1]);
    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    int pasS_length = (mrows > ncols) ? mrows : ncols;
    // mexPrintf("\n%d %d pasS dimensions.\n", mrows, ncols); 
    cVec = mxGetInt32s(prhs[2]);
    // mexPrintf(" = %d\n", cVec[0]);
    mrows = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    // mexPrintf("\n%d %d cVec dimensions.\n", mrows, ncols);
    g = mxGetScalar(prhs[3]); 
    nelx = (int) mxGetScalar(prhs[4]); 
    nely = (int) mxGetScalar(prhs[5]); 
    nelz = (int) mxGetScalar(prhs[6]); 
    direction = (int) mxGetScalar(prhs[7]);
    // mexPrintf("\n g %f \n", g);
    // mexPrintf("\nnelx %d \n", nelx);
    // mexPrintf("\nnely %d \n", nely);
    // mexPrintf("\nnelz %d \n", nelz);

    Assembly_F0_1(F0,pasS,pasS_length, cVec,g, nelx, nely, nelz, direction);
    
}

__device__ double atomicAdd1(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

void Assembly_F0_1(double*F0,double *pasS,int pasS_length, mxInt32 *cVec0,double g, int nelx, int nely, int nelz, int direction)
{

    int nnel=nelx*nely*nelz;
    int ndof = 3*(nelx + 1) * (nely + 1) * (nelz + 1);
    cudaDeviceReset();
    cudaSetDevice(0);

    int lenW=estWorkSpace(nelx, nely, nelz);
    double *devW;
    HANDLE_ERROR( cudaMalloc( (void**)&devW, (lenW+pasS_length) * sizeof(double)) );
    int *devW1;
    HANDLE_ERROR( cudaMalloc( (void**)&devW1, nnel * sizeof(int)) );
    double *W=&devW[0];
    int lW=0;
    double *F0_d=&W[lW]; lW += ndof;
    HANDLE_ERROR( cudaMemcpy( F0_d, F0, ndof * sizeof(double), cudaMemcpyHostToDevice ) );
    double *pasS_d=&W[lW]; lW += pasS_length;
    HANDLE_ERROR( cudaMemcpy( pasS_d, pasS, pasS_length * sizeof(double), cudaMemcpyHostToDevice ));

    int *cVec_d=&devW1[0];
    HANDLE_ERROR( cudaMemcpy( cVec_d, cVec0, nnel * sizeof(int), cudaMemcpyHostToDevice ));
    if (direction == 3)
    {
        Assembly_F0_device_3<<<128, 128>>>(F0_d,pasS_d,cVec_d,g,pasS_length,  nelx, nely, nelz);check();
    }
    else if (direction == 2)   
    {
        Assembly_F0_device_2<<<128, 128>>>(F0_d,pasS_d,cVec_d,g,pasS_length,  nelx, nely, nelz);check();
    }
    else if (direction == 1)
    {
        Assembly_F0_device_1<<<128, 128>>>(F0_d,pasS_d,cVec_d,g,pasS_length,  nelx, nely, nelz);check();
    }
        
    HANDLE_ERROR(cudaMemcpy(F0, F0_d, ndof * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR( cudaFree( devW ));
    HANDLE_ERROR(cudaFree(devW1)); 

}

int estWorkSpace(int nelx0, int nely0, int nelz0)
{
    int lW;
    int nnel0=nelx0*nely0*nelz0;
    int ndof = 3*(nelx0 + 1) * (nely0 + 1) * (nelz0 + 1);
    lW=ndof * 1;   
    
    return lW;
}

__global__ void Assembly_F0_device_3( double *F0_d1, double *pasS_d1,int *cVec_d1, double g,int pasS_length0, int nelx,int nely,int nelz)
{
    

	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;
    while (global_id < pasS_length0)
    {
        int pasS_int_index;
        pasS_int_index = 0;
        pasS_int_index = int (pasS_d1[global_id]);
        int gg[1];
        gg[0] = 0;
        gg[0] = cVec_d1[pasS_int_index-1];
        atomicAdd1(&F0_d1[-1 + gg[0] + 2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) - 1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] - 1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1) + 2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) + 2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) - 1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1) - 1], -0.125 * g);

        global_id += blockDim.x * gridDim.x;
    }
}


__global__ void Assembly_F0_device_2( double *F0_d1, double *pasS_d1,int *cVec_d1, double g,int pasS_length0, int nelx,int nely,int nelz)
{
    

	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;
    while (global_id < pasS_length0)
    {
        int pasS_int_index;
        pasS_int_index = 0;
        pasS_int_index = int (pasS_d1[global_id]);
        int gg[1];
        gg[0] = 0;
        gg[0] = cVec_d1[pasS_int_index-1];
        atomicAdd1(&F0_d1[-1 + gg[0] + 1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + -2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1) + 1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+1], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-2], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1) -2], -0.125 * g);

        global_id += blockDim.x * gridDim.x;
    }
}


__global__ void Assembly_F0_device_1( double *F0_d1, double *pasS_d1,int *cVec_d1, double g,int pasS_length0, int nelx,int nely,int nelz)
{
    

	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;
    while (global_id < pasS_length0)
    {
        int pasS_int_index;
        pasS_int_index = 0;
        pasS_int_index = int (pasS_d1[global_id]);
        int gg[1];
        gg[0] = 0;
        gg[0] = cVec_d1[pasS_int_index-1];
        atomicAdd1(&F0_d1[-1 + gg[0] + 0], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 0], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -3], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + -3], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1) + 0], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+0], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-3], -0.125 * g);
        atomicAdd1(&F0_d1[-1 + gg[0] + 3*(nely+1) -3], -0.125 * g);

        global_id += blockDim.x * gridDim.x;
    }
}