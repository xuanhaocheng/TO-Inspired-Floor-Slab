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


int estWorkSpace(int nelx0, int nely0, int nelz0);
__global__ void dot_product_mul(double *Md, double *c,int *cVec1,  double *r, int nelx, int nely, int  nelz);
__global__ void print_cVec_d(double *Md, int *cVec1, double *UU_d, int nelx, int nely, int  nelz);
// __global__ void su_data_print(double *Md,int *cVec1, double *UU_Aed, int nelx, int nely, int  nelz);
// __global__ void su_data_print(double *Md,int *cVec1, double *UU_Aed_sum, int nelx, int nely, int  nelz);
// __global__ void su_data_print(double *Md,int *cVec1, double *UU_Aed_sum, int nelx, int nely, int  nelz);
__global__ void su_data_print(double *Md,int *cVec1, double *dsk_d, double *UU_Aed_sum, int nelx, int nely, int  nelz);

// void sum_UKU1(double Ae0[][24],double *dsk0,double *U0, mxInt32 *cVec0,int nelx, int nely, int nelz, double *dc);
// void sum_UKU1(double Ae0[][24],double *dsk0,double *U0, mxInt32 *cVec0,int nelx, int nely, int nelz, double *UU_K);
// void sum_UKU1(double Ae0[][24],double *dsk0,double *U0, mxInt32 *cVec0,int nelx, int nely, int nelz, double *UUAe);
void sum_UKU1(double Ae0[][24],double *dsk0,double *U0, mxInt32 *cVec0,int nelx, int nely, int nelz, double *UUAe);


// void fprintII(const char *format, int v1, int v2);

// void copyArray(double *source, double *destination, size_t size) {
//     memcpy(destination, source, size * sizeof(double));
// }

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

// #define maxThreads 1024
// #define maxDofs0 20000

__constant__ double Ae[24][24];

double nu=0.3, hx0=1, hy0=1, hz0=1;


/*-----------------------------------------------------------------------*/
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *Aep, *U, *dsk;
    double Ae[24][24];
    // double *dc;
    double *UUAesum;
    mxInt32 *cVec;
    
    int nelx, nely, nelz;

    Aep = mxGetDoubles(prhs[0]);
    int mrows = mxGetM(prhs[0]);
    int ncols = mxGetN(prhs[0]);
    for (int i=0; i<24; i++)
    for (int j=0; j<24; j++)
        Ae[j][i]=Aep[24*i+j];
    
    // mexPrintf("\n%d %d input dimensions.\n", mrows, ncols);
    dsk = mxGetDoubles(prhs[1]);
    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    // mexPrintf("\n%d %d dsk dimensions.\n", mrows, ncols); 
    U = mxGetDoubles(prhs[2]);
    mrows = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    // mexPrintf("\n%d %d U dimensions.\n", mrows, ncols);
    cVec = mxGetInt32s(prhs[3]);
    // mexPrintf(" = %d\n", cVec[0]);
    mrows = mxGetM(prhs[3]);
    ncols = mxGetN(prhs[3]);
    // mexPrintf("\n%d %d cVec dimensions.\n", mrows, ncols);
    nelx = (int) mxGetScalar(prhs[4]); 
    nely = (int) mxGetScalar(prhs[5]); 
    nelz = (int) mxGetScalar(prhs[6]); 
    // mexPrintf("\nnelx %d \n", nelx);
    // mexPrintf("\nnely %d \n", nely);
    // mexPrintf("\nnelz %d \n", nelz);

    // double *dc = (double*)malloc(nelx * nely * nelz * sizeof(double));;
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    // plhs[0] = mxCreateDoubleMatrix(nelx * nely * nelz, 1,mxREAL);
    // // plhs[0] = 0.0;
    // dc = mxGetPr(plhs[0]);
    // sum_UKU1(Ae, dsk, U,cVec, nelx, nely, nelz,dc);

    // plhs[0] = mxCreateDoubleMatrix(nelx * nely * nelz, 24,mxREAL);
    // // plhs[0] = 0.0;
    // UU_K = mxGetPr(plhs[0]);
    // sum_UKU1(Ae, dsk, U,cVec, nelx, nely, nelz,UU_K);

    plhs[0] = mxCreateDoubleMatrix(nelx * nely * nelz, 1,mxREAL);
    // plhs[0] = 0.0;
    UUAesum = mxGetPr(plhs[0]);
    sum_UKU1(Ae, dsk, U,cVec, nelx, nely, nelz,UUAesum);
    
}

void sum_UKU1(double Ae0[][24],double *dsk0,double *U0, mxInt32 *cVec0,int nelx, int nely, int nelz, double *UUAe)
{

    // printf("nelx %d \n", nelx);
    // printf("nely %d \n", nely);
    // printf("nelz %d \n", nelz);
    // printf("sum_uku! \n");
    

    // const int WIDTH = 24;
    int nnel=nelx*nely*nelz;
    int ndof = 3*(nelx + 1) * (nely + 1) * (nelz + 1);
    // printf("\n ndof is  %d \n", ndof);
    const int nBlocksX = 32;
    const int nThreadsX=32;
    const int nThreadsY=32;
	
	int LENGTH = nnel * 24;
    
	int NUM = LENGTH / 24;
	const int nBlocksY = (NUM+nThreadsY*nThreadsX*nBlocksX -1) / (nThreadsY*nThreadsX*nBlocksX);
	// printf("nBlocksY %d", nBlocksY);
    dim3 grids(nBlocksX,nBlocksY);
    dim3 threads(nThreadsX,nThreadsY);

    // cudaEvent_t start, stop;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    cudaDeviceReset();
    cudaSetDevice(0);
    // cudaFree(0); // This will establish a context on the device
    // size_t f, t;
    // cudaMemGetInfo(&f, &t);
    HANDLE_ERROR( cudaMemcpyToSymbol( Ae, Ae0, sizeof(double) * 24*24) );

    int lenW=estWorkSpace(nelx, nely, nelz);
    // printf("lenW is %d \n", lenW);
    // fprintII("Reported GPU memory. Total: %d kb Free: %d kb\n",t/1024,f/1024);
    // fprintII("Number of elements %d, Device memory required %d (Mb)\n", nnel, (lenW * sizeof(double))/1048576+1);

    double *devW;
    HANDLE_ERROR( cudaMalloc( (void**)&devW, (lenW+NUM) * sizeof(double)) );
    int *devW1;
    HANDLE_ERROR( cudaMalloc( (void**)&devW1, NUM * sizeof(int)) );
    double *W=&devW[0];
    int lW=0;
    double *U0_d=&W[lW]; lW += ndof;
    HANDLE_ERROR( cudaMemcpy( U0_d, U0, ndof * sizeof(double), cudaMemcpyHostToDevice ) );
    double *dsk_d=&W[lW]; lW += NUM;
    HANDLE_ERROR( cudaMemcpy( dsk_d, dsk0, NUM * sizeof(double), cudaMemcpyHostToDevice ));
    double *dc_d=&W[lW]; lW += NUM;
    double *UU_Aed_sum=&W[lW]; lW += NUM;



    int *cVec_d=&devW1[0];
    HANDLE_ERROR( cudaMemcpy( cVec_d, cVec0, NUM * sizeof(int), cudaMemcpyHostToDevice ));
    // printf("\n U0 \n");
    // for(int i = 0; i < 20; ++i)
    // {
    //     printf("%f ",U0[i]);
    // }
    // printf("\n dsk \n");
    // for(int i = 0; i < 20; ++i)
    // {
    //     printf("%f ",dsk0[i]);
    // }
    // printf("\n Ae \n");
    // for(int i = 0; i < 24; ++i)
        
    // {
    //     for(int j = 0; j < 24; ++j)
    //     {
    //         printf("%f ",Ae0[i][j]);
    //     }
    //     ("\n");
    // }
    // printf("\n cVec \n");
    // for(int i = 0; i < nelx* nely* nelz; ++i)
        
    // {
        
    //     printf("%d ",cVec0[i]);
        
    // }
    // printf("\n");

    // double* UU_Ae = (double*)malloc(LENGTH * sizeof(double));

    // cudaEventRecord(start, 0);
    su_data_print<<<128, 128>>>(U0_d,cVec_d,dsk_d,UU_Aed_sum, nelx, nely, nelz);check();
    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf(" elapsedTime %f\n", elapsedTime);




    // su_data_print<<<8, 8>>>(U0_d,cVec_d,UU_Aed, nelx, nely, nelz);check();
    HANDLE_ERROR(cudaMemcpy(UUAe, UU_Aed_sum, NUM * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR( cudaFree( devW ));
    HANDLE_ERROR(cudaFree(devW1)); 





    // double* UU = (double*)malloc(LENGTH * sizeof(double));
    // print_cVec_d<<<grids, threads>>>(U0_d,cVec_d,UU_d, nelx, nely, nelz);check();
    // HANDLE_ERROR(cudaMemcpy(UU_K, UU_d, LENGTH * sizeof(double), cudaMemcpyDeviceToHost));
    // HANDLE_ERROR( cudaFree( devW ));
    // HANDLE_ERROR(cudaFree(devW1));
    // for (int i = 0; i < 100; ++i)
    // {
    //     printf("%f ", UU[i]);
    //     // printf("%f ", UU[nnel - 1 - i]);
    // }

    // dot_product_mul<<<grids, threads>>>(U0_d, dsk_d,cVec_d, dc_d,nelx, nely, nelz);check();
    // // double* result_dc = (double*)malloc(nnel * sizeof(double));
    // // double *dc1 = (double*)malloc(NUM * sizeof(double));
    // HANDLE_ERROR(cudaMemcpy(dc, dc_d, nnel * sizeof(double), cudaMemcpyDeviceToHost));
    // HANDLE_ERROR( cudaFree( devW ));
    // HANDLE_ERROR(cudaFree(devW1));
    
    // printf("\n dc1 \n");
    // dc = result_dc;
    // copyArray(result_dc, dc, sizeof(dc));
    // for (int i = 0; i < 100; ++i)
    // {
    //     printf("%f ", dc1[i]);
    //     printf("%f ", dc1[nnel - 1 - i]);
    // }
    
    
}

int estWorkSpace(int nelx0, int nely0, int nelz0)
{
    int lW;
    int nnel0=nelx0*nely0*nelz0;
    int ndof = 3*(nelx0 + 1) * (nely0 + 1) * (nelz0 + 1);
    lW=ndof * 1 + nnel0 * 1 + nnel0 * 1;   
    
    return lW;
}

__global__ void dot_product_mul(double *Md, double *c,int *cVec1,  double *r, int nelx, int nely, int  nelz)
{
    

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
	
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int global_id = tid + bid * (blockDim.x * blockDim.y);
    // int total_thread_num = HEIGHT;
	double suData[24];
    double sData[24];
    double gData[1];
    int gg[1]; 
    // int row = global_id;
    // sData[24] = 0.0;
    for (int i = 0; i < 24; ++i) 
    {
        sData[i] = 0.0;
        suData[i] = 0.0;
    }
    gData[0] = 0.0;
    gg[0] = 0;
    gg[0] = cVec1[global_id];
    sData[0] = Md[-1 + gg[0] + 0];
    sData[1] = Md[-1 + gg[0] + 1];
    sData[2] = Md[-1 + gg[0] + 2];
    sData[3] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 0];
    sData[4] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 1];
    sData[5] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2];
    sData[6] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -3];
    sData[7] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -2];
    sData[8] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -1];
    sData[9] = Md[-1 + gg[0] + -3];
    sData[10] = Md[-1 + gg[0] + -2];
    sData[11] = Md[-1 + gg[0] + -1];
    sData[12] = Md[-1 + gg[0] + 3*(nely+1) + 0];
    sData[13] = Md[-1 + gg[0] + 3*(nely+1) + 1];
    sData[14] = Md[-1 + gg[0] + 3*(nely+1) + 2];
    sData[15] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+0];
    sData[16] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+1];
    sData[17] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+2];
    sData[18] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-3];
    sData[19] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-2];
    sData[20] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-1];
    sData[21] = Md[-1 + gg[0] + 3*(nely+1) -3];
    sData[22] = Md[-1 + gg[0] + 3*(nely+1) -2];
    sData[23] = Md[-1 + gg[0] + 3*(nely+1) -1];         



    for(int col = 0; col < 24 ; ++col)
    {
        for(int k = 0; k< 24; ++k)
        {
            // ssData[col]+= Md[k*HEIGHT+row]*Ae[k][col];
            suData[col]+= sData[k]*Ae[k][col];

        } 
    }
    __syncthreads();
	// int cc = 0;
    for(int  i=0; i < 24; ++i)
    {
        gData[0] += sData[i] * suData[i];
    }
	// while(global_id < HEIGHT* 24)
	// 	{
	// 		gData[0] += sData[cc] * ssData[cc];
	// 		global_id += total_thread_num;
    //         cc += 1;			
	// 	}
	// 	__syncthreads();
    __syncthreads();

	r[global_id] = gData[0] * c[global_id];

}



// __global__ void print_cVec_d(double *Md, int *cVec1, double *UU_d, int nelx, int nely, int  nelz)
// {
    

//     int tid = threadIdx.x + threadIdx.y * blockDim.x;
	
// 	int bid = blockIdx.x + blockIdx.y * gridDim.x;
// 	int global_id = tid + bid * (blockDim.x * blockDim.y);
//     // int total_thread_num = HEIGHT;
// 	double suData[24];
//     double sData[24];
//     double gData[1];
//     int gg[1]; 
//     // int row = global_id;
//     // sData[24] = 0.0;
//     for (int i = 0; i < 24; ++i) 
//     {
//         sData[i] = 0.0;
//         suData[i] = 0.0;
//     }
//     gData[0] = 0.0;
//     gg[0] = 0;
//     gg[0] = cVec1[global_id];
//     sData[0] = Md[-1 + gg[0] + 0];
//     sData[1] = Md[-1 + gg[0] + 1];
//     sData[2] = Md[-1 + gg[0] + 2];
//     sData[3] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 0];
//     sData[4] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 1];
//     sData[5] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2];
//     sData[6] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -3];
//     sData[7] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -2];
//     sData[8] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -1];
//     sData[9] = Md[-1 + gg[0] + -3];
//     sData[10] = Md[-1 + gg[0] + -2];
//     sData[11] = Md[-1 + gg[0] + -1];
//     sData[12] = Md[-1 + gg[0] + 3*(nely+1) + 0];
//     sData[13] = Md[-1 + gg[0] + 3*(nely+1) + 1];
//     sData[14] = Md[-1 + gg[0] + 3*(nely+1) + 2];
//     sData[15] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+0];
//     sData[16] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+1];
//     sData[17] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+2];
//     sData[18] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-3];
//     sData[19] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-2];
//     sData[20] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-1];
//     sData[21] = Md[-1 + gg[0] + 3*(nely+1) -3];
//     sData[22] = Md[-1 + gg[0] + 3*(nely+1) -2];
//     sData[23] = Md[-1 + gg[0] + 3*(nely+1) -1];   
//     int nnel1 = nelx * nely * nelz;
//     for(int i = 0; i < 24; ++i)
//     {
//         UU_d[global_id + nnel1 * i ] = sData[i];
//     }
//     // if (global_id == 0)   
//     // {
//     //     printf("\n Ucmat \n");
//     //     for (int i = 0; i < 24; ++i)
//     //     {
//     //         printf("%f ", sData[i]);
//     //     }
//     // }

// }


__global__ void su_data_print(double *Md,int *cVec1, double *dsk_d, double *UU_Aed_sum, int nelx, int nely, int  nelz)
{
    

    // int tid = threadIdx.y + threadIdx.x * blockDim.y;
	
	// int bid = blockIdx.y + blockIdx.x * gridDim.y;
	int global_id = threadIdx.x + blockIdx.x *  blockDim.x;

    // int ix = blockIdx.x * blockDim.x + threadIdx.x;
	
	// int iy = blockIdx.y *blockDim.y + threadIdx.y;
	// int global_id = iy * 4 + ix;
    // mexPrintf("\n global_id is %d \n", global_id);
    while (global_id < nelx * nely * nelz)
    {
        // int total_thread_num = HEIGHT;
        double suData[24];
        double sData[24];
        double gData[1];
        int gg[1]; 
        // int row = global_id;
        // sData[24] = 0.0;
        for (int i = 0; i < 24; ++i) 
        {
            sData[i] = 0.0;
            suData[i] = 0.0;
        }
        gData[0] = 0.0;
        int nnel1 = nelx * nely * nelz;
        gg[0] = 0;
        gg[0] = cVec1[global_id];
        sData[0] = Md[-1 + gg[0] + 0];
        sData[1] = Md[-1 + gg[0] + 1];
        sData[2] = Md[-1 + gg[0] + 2];
        sData[3] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 0];
        sData[4] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 1];
        sData[5] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + 2];
        sData[6] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -3];
        sData[7] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -2];
        sData[8] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+1) + -1];
        sData[9] = Md[-1 + gg[0] + -3];
        sData[10] = Md[-1 + gg[0] + -2];
        sData[11] = Md[-1 + gg[0] + -1];
        sData[12] = Md[-1 + gg[0] + 3*(nely+1) + 0];
        sData[13] = Md[-1 + gg[0] + 3*(nely+1) + 1];
        sData[14] = Md[-1 + gg[0] + 3*(nely+1) + 2];
        sData[15] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+0];
        sData[16] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+1];
        sData[17] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+2];
        sData[18] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-3];
        sData[19] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-2];
        sData[20] = Md[-1 + gg[0] + 3*(nely+1)*(nelz+2)+-1];
        sData[21] = Md[-1 + gg[0] + 3*(nely+1) -3];
        sData[22] = Md[-1 + gg[0] + 3*(nely+1) -2];
        sData[23] = Md[-1 + gg[0] + 3*(nely+1) -1];         



        for(int col = 0; col < 24 ; ++col)
        {
            for(int k = 0; k< 24; ++k)
            {
                // ssData[col]+= Md[k*HEIGHT+row]*Ae[k][col];
                suData[col]+= sData[k]*Ae[k][col];

            } 
        }

        

        for(int  i=0; i < 24; ++i)
        {
            gData[0] += sData[i] * suData[i];
        }

    
        UU_Aed_sum[global_id] = gData[0] * dsk_d[global_id];
        // UU_Aed_sum[global_id] = global_id;
        global_id += blockDim.x * gridDim.x;
    }
    
}