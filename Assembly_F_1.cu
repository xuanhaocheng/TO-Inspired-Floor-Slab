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
// #include <device_launch_parameters.h>



int estWorkSpace(int nelx0, int nely0, int nelz0);
__global__ void Assembly_F_device_3( double *F_d1, double *act_d1,double *xPhys0_d1, int *cVec_d1, double g,int act_length0, int nelx,int nely,int nelz);
__global__ void Assembly_F_device_2( double *F_d1, double *act_d1,double *xPhys0_d1, int *cVec_d1, double g,int act_length0, int nelx,int nely,int nelz);
__global__ void Assembly_F_device_1( double *F_d1, double *act_d1,double *xPhys0_d1, int *cVec_d1, double g,int act_length0, int nelx,int nely,int nelz);
__device__ double atomicAdd1(double* address, double val);
void Assembly_F_1(double*F,double *act,int act_length,double *xPhys0, mxInt32 *cVec0,double g, int nelx, int nely, int nelz, int direction);

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
    double *F, *act,*xPhys;
    mxInt32 *cVec;
    
    int nelx, nely, nelz,direction;
    double g;

    F = mxGetDoubles(prhs[0]);
    int mrows = mxGetM(prhs[0]);
    int ncols = mxGetN(prhs[0]);
    // mexPrintf("\n F %d %d input dimensions.\n", mrows, ncols);
    act = mxGetDoubles(prhs[1]);
    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    int act_length = (mrows > ncols) ? mrows : ncols;
    // mexPrintf("\n%d %d act dimensions.\n", mrows, ncols); 
    xPhys = mxGetDoubles(prhs[2]);
    mrows = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    // mexPrintf("\n%d %d xphys dimensions.\n", mrows, ncols); 
    cVec = mxGetInt32s(prhs[3]);
    // mexPrintf(" = %d\n", cVec[0]);
    mrows = mxGetM(prhs[3]);
    ncols = mxGetN(prhs[3]);
    // mexPrintf("\n%d %d cVec dimensions.\n", mrows, ncols);
    g = mxGetScalar(prhs[4]); 
    nelx = (int) mxGetScalar(prhs[5]); 
    nely = (int) mxGetScalar(prhs[6]); 
    nelz = (int) mxGetScalar(prhs[7]); 
    direction = (int) mxGetScalar(prhs[8]);
    // mexPrintf("\n g %f \n", g);
    // mexPrintf("\nnelx %d \n", nelx);
    // mexPrintf("\nnely %d \n", nely);
    // mexPrintf("\nnelz %d \n", nelz);

    Assembly_F_1(F,act,act_length,xPhys, cVec,g, nelx, nely, nelz,direction);
    
}

// #if __CUDA_ARCH__ < 600
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
// #endif

void Assembly_F_1(double*F,double *act,int act_length,double *xPhys0, mxInt32 *cVec0,double g, int nelx, int nely, int nelz, int direction)
{

    int nnel=nelx*nely*nelz;
    int ndof = 3*(nelx + 1) * (nely + 1) * (nelz + 1);
    // printf("\n xphys[1432910] is %0.52f", xPhys0[1432909]);
    cudaDeviceReset();
    cudaSetDevice(0);

    int lenW=estWorkSpace(nelx, nely, nelz);
    double *devW;
    HANDLE_ERROR( cudaMalloc( (void**)&devW, (lenW+act_length) * sizeof(double)) );
    int *devW1;
    HANDLE_ERROR( cudaMalloc( (void**)&devW1, nnel * sizeof(int)) );
    double *W=&devW[0];
    int lW=0;
    double *F_d=&W[lW]; lW += ndof;
    HANDLE_ERROR( cudaMemcpy( F_d, F, ndof * sizeof(double), cudaMemcpyHostToDevice ) );
    double *xPhys0_d=&W[lW]; lW += nnel;
    HANDLE_ERROR( cudaMemcpy( xPhys0_d, xPhys0, nnel * sizeof(double), cudaMemcpyHostToDevice ) );
    double *act_d=&W[lW]; lW += act_length;
    HANDLE_ERROR( cudaMemcpy( act_d, act, act_length * sizeof(double), cudaMemcpyHostToDevice ));

    int *cVec_d=&devW1[0];
    HANDLE_ERROR( cudaMemcpy( cVec_d, cVec0, nnel * sizeof(int), cudaMemcpyHostToDevice ));
    if (direction == 3)
    {
        Assembly_F_device_3<<<1, 1024>>>(F_d,act_d,xPhys0_d,cVec_d,g,act_length,  nelx, nely, nelz);check();
    }
    else if (direction == 2)   
    {
        Assembly_F_device_2<<<1, 1024>>>(F_d,act_d,xPhys0_d,cVec_d,g,act_length,  nelx, nely, nelz);check();
    }
    else if (direction == 1)
    {
        Assembly_F_device_1<<<1, 1024>>>(F_d,act_d,xPhys0_d,cVec_d,g,act_length,  nelx, nely, nelz);check();
    }
    HANDLE_ERROR(cudaMemcpy(F, F_d, ndof * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR( cudaFree( devW ));
    HANDLE_ERROR(cudaFree(devW1)); 

}

int estWorkSpace(int nelx0, int nely0, int nelz0)
{
    int lW;
    int nnel0=nelx0*nely0*nelz0;
    int ndof = 3*(nelx0 + 1) * (nely0 + 1) * (nelz0 + 1);
    lW=ndof * 1 + nnel0 * 1;   
    
    return lW;
}

__global__ void Assembly_F_device_3( double *F_d1, double *act_d1,double *xPhys0_d1, int *cVec_d1, double g,int act_length0, int nelx,int nely,int nelz)
{
    

	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;
    while (global_id < act_length0)
    {
        int act_int_index;
        act_int_index = 0;
        act_int_index = int(act_d1[global_id]);
        int gg[1];
        gg[0] = 0;
        gg[0] = cVec_d1[act_int_index-1];
        double aa[1];
        aa[0] = 0.0;
        aa[0] = xPhys0_d1[act_int_index - 1];
        double bb = -aa[0] * 0.125 * g;
        atomicAdd1(&F_d1[-1 + gg[0] + 2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) - 1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] - 1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) + 2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) + 2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) - 1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) - 1], bb);
        __syncthreads();
       
        global_id += blockDim.x * gridDim.x;
    }
}

__global__ void Assembly_F_device_2( double *F_d1, double *act_d1,double *xPhys0_d1, int *cVec_d1, double g,int act_length0, int nelx,int nely,int nelz)
{
    

	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;
    while (global_id < act_length0)
    {
        int act_int_index;
        act_int_index = 0;
        act_int_index = int(act_d1[global_id]);
        int gg[1];
        gg[0] = 0;
        gg[0] = cVec_d1[act_int_index-1];
        double aa[1];
        aa[0] = 0.0;
        aa[0] = xPhys0_d1[act_int_index - 1];
        double bb = -aa[0] * 0.125 * g;
        // atomicAdd1(&F_d1[-1 + gg[0] + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) - 1], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] - 1], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) - 1], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) - 1], bb);

        atomicAdd1(&F_d1[-1 + gg[0] + 1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + -2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) + 1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+1], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-2], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) -2], bb);

        __syncthreads();
       
        global_id += blockDim.x * gridDim.x;
    }
}

__global__ void Assembly_F_device_1( double *F_d1, double *act_d1,double *xPhys0_d1, int *cVec_d1, double g,int act_length0, int nelx,int nely,int nelz)
{
    

	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;
    while (global_id < act_length0)
    {
        int act_int_index;
        act_int_index = 0;
        act_int_index = int(act_d1[global_id]);
        int gg[1];
        gg[0] = 0;
        gg[0] = cVec_d1[act_int_index-1];
        double aa[1];
        aa[0] = 0.0;
        aa[0] = xPhys0_d1[act_int_index - 1];
        double bb = -aa[0] * 0.125 * g;
        // atomicAdd1(&F_d1[-1 + gg[0] + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) - 1], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] - 1], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) + 2], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2) - 1], bb);
        // atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) - 1], bb);

        atomicAdd1(&F_d1[-1 + gg[0] + 0], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 0], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -3], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + -3], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) + 0], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+0], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-3], bb);
        atomicAdd1(&F_d1[-1 + gg[0] + 3*(nely+1) -3], bb);

        __syncthreads();
       
        global_id += blockDim.x * gridDim.x;
    }
}