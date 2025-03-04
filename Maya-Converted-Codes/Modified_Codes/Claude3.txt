#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <omp.h>  // Added for CPU multithreading

// Constants
#define BLOCK_SIZE 256
#define GAMMA 1.4
#define RD double
#define MAX_FILENAME 256
#define HALO_SIZE 2  // Increased halo size for better overlap
#define USE_SHARED_MEMORY 1
#define USE_PINNED_MEMORY 1
#define USE_MIXED_PRECISION 0  // Set to 1 to enable mixed precision

// Error checking macro for CUDA calls with file and line info
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer macro for performance measurement
#define TIME_FUNCTION(label, func) \
do { \
    cudaEvent_t start, stop; \
    float milliseconds = 0; \
    CUDA_CHECK(cudaEventCreate(&start)); \
    CUDA_CHECK(cudaEventCreate(&stop)); \
    CUDA_CHECK(cudaEventRecord(start)); \
    func; \
    CUDA_CHECK(cudaEventRecord(stop)); \
    CUDA_CHECK(cudaEventSynchronize(stop)); \
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop)); \
    printf("%s: %.3f ms\n", label, milliseconds); \
    CUDA_CHECK(cudaEventDestroy(start)); \
    CUDA_CHECK(cudaEventDestroy(stop)); \
} while(0)

// Structure to hold the grid section for each GPU
typedef struct {
    int startIdx;
    int endIdx;
    int localSize;
    int device;
    int haloStartL;  // Left halo start index (global)
    int haloEndL;    // Left halo end index (global)
    int haloStartR;  // Right halo start index (global)
    int haloEndR;    // Right halo end index (global)
    cudaStream_t computeStream;
    cudaStream_t transferStreamLeft;  // Separate streams for left and right transfers
    cudaStream_t transferStreamRight;
    cudaEvent_t computeDone;
    cudaEvent_t leftTransferDone;  // Separate events for better synchronization
    cudaEvent_t rightTransferDone;
} GPUSection;

// Structure for solution variables (SoA layout for better memory access)
typedef struct {
    double *rho;     // Density
    double *rhoU;    // Momentum
    double *rhoE;    // Energy
} SolutionVariables;

// Mixed precision version for computation
#if USE_MIXED_PRECISION
typedef struct {
    float *rho;      // Density
    float *rhoU;     // Momentum
    float *rhoE;     // Energy
} SolutionVariablesSP;
#endif

// Device constants
__constant__ double d_gamma;
__constant__ double d_dx;
__constant__ double d_dt;
__constant__ double d_CFL;

// CUDA kernel for Roe solver interfaces with shared memory optimization
__global__ void interfacesKernel(double *d_rho, double *d_rhoU, double *d_rhoE,
                              double *d_Favg_rho, double *d_Favg_rhoU, double *d_Favg_rhoE,
                              double *d_Diss_rho, double *d_Diss_rhoU, double *d_Diss_rhoE,
                              int startIdx, int endIdx) {
#if USE_SHARED_MEMORY
    // Shared memory for caching density, momentum, and energy
    __shared__ double s_rho[BLOCK_SIZE + 1];
    __shared__ double s_rhoU[BLOCK_SIZE + 1];
    __shared__ double s_rhoE[BLOCK_SIZE + 1];
    
    // Local thread ID within the block
    int tid = threadIdx.x;
    // Global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    // Load data into shared memory
    if (i < endIdx + 1) {  // +1 to handle the last interface
        s_rho[tid] = d_rho[i];
        s_rhoU[tid] = d_rhoU[i];
        s_rhoE[tid] = d_rhoE[i];
    }
    
    // Load extra element needed for the last thread in the block
    if (tid == BLOCK_SIZE - 1 || i + 1 == endIdx + 1) {
        s_rho[tid + 1] = d_rho[i + 1];
        s_rhoU[tid + 1] = d_rhoU[i + 1];
        s_rhoE[tid + 1] = d_rhoE[i + 1];
    }
    
    __syncthreads();
#else
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
#endif
    
    if (i >= startIdx && i < endIdx) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Defining Roe Variables from Q
#if USE_SHARED_MEMORY
        RhoL = s_rho[threadIdx.x];
        RhoR = s_rho[threadIdx.x + 1];
        double rhoUL = s_rhoU[threadIdx.x];
        double rhoUR = s_rhoU[threadIdx.x + 1];
        UL = rhoUL / RhoL;
        UR = rhoUR / RhoR;
        double rhoEL = s_rhoE[threadIdx.x];
        double rhoER = s_rhoE[threadIdx.x + 1];
        EL = rhoEL / RhoL;
        ER = rhoER / RhoR;
#else
        RhoL = d_rho[i];
        RhoR = d_rho[i+1];
        UL = d_rhoU[i] / RhoL;
        UR = d_rhoU[i+1] / RhoR;
        EL = d_rhoE[i] / RhoL;
        ER = d_rhoE[i+1] / RhoR;
#endif
        
        // Fast math for better GPU performance
        PL = (RhoL * EL - 0.5 * RhoL * (UL * UL)) * (GAMMA - 1.0);
        PR = (RhoR * ER - 0.5 * RhoR * (UR * UR)) * (GAMMA - 1.0);
        HL = EL + PL / RhoL;
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg_rho[i] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg_rhoU[i] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg_rhoE[i] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
        // Roe Variables At Interface - using fastmath intrinsics where possible
        Rhoroe = sqrt(RhoL * RhoR);
        double sqrtRhoL = sqrt(RhoL);
        double sqrtRhoR = sqrt(RhoR);
        double invSumSqrt = 1.0 / (sqrtRhoL + sqrtRhoR);
        
        Uroe = (sqrtRhoL * UL + sqrtRhoR * UR) * invSumSqrt;
        Hroe = (sqrtRhoL * HL + sqrtRhoR * HR) * invSumSqrt;
        Croe = sqrt((GAMMA - 1.0) * (Hroe - 0.5 * (Uroe * Uroe))); 
        
        // Eigenvalues
        Lambda[0] = Uroe - Croe;
        Lambda[1] = Uroe;
        Lambda[2] = Uroe + Croe;
        
        // Eigenvectors
        Eig[0][0] = 1.0;
        Eig[0][1] = Uroe - Croe;
        Eig[0][2] = Hroe - Uroe * Croe;
        
        Eig[1][0] = 1.0;
        Eig[1][1] = Uroe;
        Eig[1][2] = 0.5 * Uroe * Uroe;
        
        Eig[2][0] = 1.0;
        Eig[2][1] = Uroe + Croe;
        Eig[2][2] = Hroe + Uroe * Croe;
        
        // Optimize Alpha coefficient calculations
        double invCroe = 1.0 / Croe;
        double invCroe2 = invCroe * invCroe;
        double UoverC = Uroe * invCroe;
        double gamma_m1 = GAMMA - 1.0;
        double rhoDiff = RhoR - RhoL;
        double rhoUDiff = RhoR * UR - RhoL * UL;
        double rhoEDiff = RhoR * ER - RhoL * EL;
        
        Alpha[0] = (0.25 * UoverC) * rhoDiff * (2.0 + gamma_m1 * UoverC) -
                   (0.5 * invCroe) * (1.0 + gamma_m1 * UoverC) * rhoUDiff +
                   0.5 * gamma_m1 * rhoEDiff * invCroe2;
        
        Alpha[1] = (1.0 - 0.5 * UoverC * UoverC * gamma_m1) * rhoDiff +
                   gamma_m1 * Uroe * rhoUDiff * invCroe2 -
                   gamma_m1 * rhoEDiff * invCroe2;
        
        Alpha[2] = -(0.25 * UoverC) * (2.0 - gamma_m1 * UoverC) * rhoDiff +
                   (0.5 * invCroe) * (1.0 - gamma_m1 * UoverC) * rhoUDiff +
                   0.5 * gamma_m1 * rhoEDiff * invCroe2;
        
        // Calculate dissipation with optimized loop and accumulation
        double diss_rho = 0.0, diss_rhoU = 0.0, diss_rhoE = 0.0;
        
        // Unroll the small loop for better performance
        double absLambda0 = fabs(Lambda[0]);
        double absLambda1 = fabs(Lambda[1]);
        double absLambda2 = fabs(Lambda[2]);
        
        diss_rho += absLambda0 * Alpha[0] * Eig[0][0];
        diss_rho += absLambda1 * Alpha[1] * Eig[1][0];
        diss_rho += absLambda2 * Alpha[2] * Eig[2][0];
        
        diss_rhoU += absLambda0 * Alpha[0] * Eig[0][1];
        diss_rhoU += absLambda1 * Alpha[1] * Eig[1][1];
        diss_rhoU += absLambda2 * Alpha[2] * Eig[2][1];
        
        diss_rhoE += absLambda0 * Alpha[0] * Eig[0][2];
        diss_rhoE += absLambda1 * Alpha[1] * Eig[1][2];
        diss_rhoE += absLambda2 * Alpha[2] * Eig[2][2];
        
        d_Diss_rho[i] = diss_rho;
        d_Diss_rhoU[i] = diss_rhoU;
        d_Diss_rhoE[i] = diss_rhoE;
    }
}

// CUDA kernel for supersonic inflow boundary condition
__global__ void superSonicInflowBCKernel(double *d_rho, double *d_rhoU, double *d_rhoE,
                                      double *d_Favg_rho, double *d_Favg_rhoU, double *d_Favg_rhoE,
                                      double *d_Diss_rho, double *d_Diss_rhoU, double *d_Diss_rhoE,
                                      int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Define constant inflow conditions for better locality
        RhoL = 1.0;
        UL = 2.95;
        double UL2 = UL * UL;
        PL = 1.0/GAMMA;
        EL = (1.0/GAMMA)/(GAMMA-1) + 0.5 * UL2;
        HL = EL + PL / RhoL;
        
        // Get right state
        RhoR = d_rho[idx+1];
        UR = d_rhoU[idx+1] / RhoR;
        ER = d_rhoE[idx+1] / RhoR;
        PR = (RhoR * ER - 0.5 * RhoR * (UR * UR)) * (GAMMA - 1.0);
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg_rho[idx] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg_rhoU[idx] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * UL2 + PL);
        d_Favg_rhoE[idx] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
        // Roe Variables At Interface
        Rhoroe = sqrt(RhoL * RhoR);
        double sqrtRhoL = sqrt(RhoL);
        double sqrtRhoR = sqrt(RhoR);
        double invSumSqrt = 1.0 / (sqrtRhoL + sqrtRhoR);
        
        Uroe = (sqrtRhoL * UL + sqrtRhoR * UR) * invSumSqrt;
        Hroe = (sqrtRhoL * HL + sqrtRhoR * HR) * invSumSqrt;
        Croe = sqrt((GAMMA - 1.0) * (Hroe - 0.5 * (Uroe * Uroe))); 
        
        // Eigenvalues
        Lambda[0] = Uroe - Croe;
        Lambda[1] = Uroe;
        Lambda[2] = Uroe + Croe;
        
        // Eigenvectors
        Eig[0][0] = 1.0;
        Eig[0][1] = Uroe - Croe;
        Eig[0][2] = Hroe - Uroe * Croe;
        
        Eig[1][0] = 1.0;
        Eig[1][1] = Uroe;
        Eig[1][2] = 0.5 * Uroe * Uroe;
        
        Eig[2][0] = 1.0;
        Eig[2][1] = Uroe + Croe;
        Eig[2][2] = Hroe + Uroe * Croe;
        
        // Optimize Alpha coefficient calculations
        double invCroe = 1.0 / Croe;
        double invCroe2 = invCroe * invCroe;
        double UoverC = Uroe * invCroe;
        double gamma_m1 = GAMMA - 1.0;
        double rhoDiff = RhoR - RhoL;
        double rhoUDiff = RhoR * UR - RhoL * UL;
        double rhoEDiff = RhoR * ER - RhoL * EL;
        
        Alpha[0] = (0.25 * UoverC) * rhoDiff * (2.0 + gamma_m1 * UoverC) -
                   (0.5 * invCroe) * (1.0 + gamma_m1 * UoverC) * rhoUDiff +
                   0.5 * gamma_m1 * rhoEDiff * invCroe2;
        
        Alpha[1] = (1.0 - 0.5 * UoverC * UoverC * gamma_m1) * rhoDiff +
                   gamma_m1 * Uroe * rhoUDiff * invCroe2 -
                   gamma_m1 * rhoEDiff * invCroe2;
        
        Alpha[2] = -(0.25 * UoverC) * (2.0 - gamma_m1 * UoverC) * rhoDiff +
                   (0.5 * invCroe) * (1.0 - gamma_m1 * UoverC) * rhoUDiff +
                   0.5 * gamma_m1 * rhoEDiff * invCroe2;
        
        // Calculate dissipation with optimized accumulation
        double diss_rho = 0.0, diss_rhoU = 0.0, diss_rhoE = 0.0;
        
        // Unroll the small loop
        double absLambda0 = fabs(Lambda[0]);
        double absLambda1 = fabs(Lambda[1]);
        double absLambda2 = fabs(Lambda[2]);
        
        diss_rho += absLambda0 * Alpha[0] * Eig[0][0];
        diss_rho += absLambda1 * Alpha[1] * Eig[1][0];
        diss_rho += absLambda2 * Alpha[2] * Eig[2][0];
        
        diss_rhoU += absLambda0 * Alpha[0] * Eig[0][1];
        diss_rhoU += absLambda1 * Alpha[1] * Eig[1][1];
        diss_rhoU += absLambda2 * Alpha[2] * Eig[2][1];
        
        diss_rhoE += absLambda0 * Alpha[0] * Eig[0][2];
        diss_rhoE += absLambda1 * Alpha[1] * Eig[1][2];
        diss_rhoE += absLambda2 * Alpha[2] * Eig[2][2];
        
        d_Diss_rho[idx] = diss_rho;
        d_Diss_rhoU[idx] = diss_rhoU;
        d_Diss_rhoE[idx] = diss_rhoE;
    }
}

// CUDA kernel for updating solution with improved memory access pattern
__global__ void updateSolutionKernel(double *d_rho, double *d_rhoU, double *d_rhoE,
                                    double *d_rho_new, double *d_rhoU_new, double *d_rhoE_new,
                                    double *d_Favg_rho, double *d_Favg_rhoU, double *d_Favg_rhoE,
                                    double *d_Diss_rho, double *d_Diss_rhoU, double *d_Diss_rhoE,
                                    int startIdx, int endIdx, double dt, double dx) {
#if USE_SHARED_MEMORY
    __shared__ double s_Favg_rho[BLOCK_SIZE + 1];
    __shared__ double s_Favg_rhoU[BLOCK_SIZE + 1];
    __shared__ double s_Favg_rhoE[BLOCK_SIZE + 1];
    __shared__ double s_Diss_rho[BLOCK_SIZE + 1];
    __shared__ double s_Diss_rhoU[BLOCK_SIZE + 1];
    __shared__ double s_Diss_rhoE[BLOCK_SIZE + 1];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    // Load current and previous values into shared memory
    if (i >= startIdx && i <= endIdx) {
        int localIdx = i - blockIdx.x * blockDim.x - startIdx;
        s_Favg_rho[localIdx] = d_Favg_rho[i-1];
        s_Favg_rhoU[localIdx] = d_Favg_rhoU[i-1];
        s_Favg_rhoE[localIdx] = d_Favg_rhoE[i-1];
        s_Diss_rho[localIdx] = d_Diss_rho[i-1];
        s_Diss_rhoU[localIdx] = d_Diss_rhoU[i-1];
        s_Diss_rhoE[localIdx] = d_Diss_rhoE[i-1];
    }
    
    if (tid == BLOCK_SIZE - 1 || i == endIdx) {
        s_Favg_rho[tid + 1] = d_Favg_rho[i];
        s_Favg_rhoU[tid + 1] = d_Favg_rhoU[i];
        s_Favg_rhoE[tid + 1] = d_Favg_rhoE[i];
        s_Diss_rho[tid + 1] = d_Diss_rho[i];
        s_Diss_rhoU[tid + 1] = d_Diss_rhoU[i];
        s_Diss_rhoE[tid + 1] = d_Diss_rhoE[i];
    }
    
    __syncthreads();
#endif
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    if (i >= startIdx && i < endIdx) {
        // Precompute the time step factor
        double dt_dx = dt / dx;
        
#if USE_SHARED_MEMORY
        // Get values from shared memory for better performance
        double favg_rho_i = s_Favg_rho[tid + 1];
        double favg_rhoU_i = s_Favg_rhoU[tid + 1];
        double favg_rhoE_i = s_Favg_rhoE[tid + 1];
        double diss_rho_i = s_Diss_rho[tid + 1];
        double diss_rhoU_i = s_Diss_rhoU[tid + 1];
        double diss_rhoE_i = s_Diss_rhoE[tid + 1];
        
        double favg_rho_im1 = s_Favg_rho[tid];
        double favg_rhoU_im1 = s_Favg_rhoU[tid];
        double favg_rhoE_im1 = s_Favg_rhoE[tid];
        double diss_rho_im1 = s_Diss_rho[tid];
        double diss_rhoU_im1 = s_Diss_rhoU[tid];
        double diss_rhoE_im1 = s_Diss_rhoE[tid];
#else
        // Get values directly from global memory
        double favg_rho_i = d_Favg_rho[i];
        double favg_rhoU_i = d_Favg_rhoU[i];
        double favg_rhoE_i = d_Favg_rhoE[i];
        double diss_rho_i = d_Diss_rho[i];
        double diss_rhoU_i = d_Diss_rhoU[i];
        double diss_rhoE_i = d_Diss_rhoE[i];
        
        double favg_rho_im1 = d_Favg_rho[i-1];
        double favg_rhoU_im1 = d_Favg_rhoU[i-1];
        double favg_rhoE_im1 = d_Favg_rhoE[i-1];
        double diss_rho_im1 = d_Diss_rho[i-1];
        double diss_rhoU_im1 = d_Diss_rhoU[i-1];
        double diss_rhoE_im1 = d_Diss_rhoE[i-1];
#endif
        
        // Update density - compute flux differences first
        double flux_rho = (favg_rho_i - 0.5 * diss_rho_i) - (favg_rho_im1 - 0.5 * diss_rho_im1);
        double flux_rhoU = (favg_rhoU_i - 0.5 * diss_rhoU_i) - (favg_rhoU_im1 - 0.5 * diss_rhoU_im1);
        double flux_rhoE = (favg_rhoE_i - 0.5 * diss_rhoE_i) - (favg_rhoE_im1 - 0.5 * diss_rhoE_im1);
        
        // Update solution variables
        d_rho_new[i] = d_rho[i] - dt_dx * flux_rho;
        d_rhoU_new[i] = d_rhoU[i] - dt_dx * flux_rhoU;
        d_rhoE_new[i] = d_rhoE[i] - dt_dx * flux_rhoE;
    }
}

// Function to calculate CFL-based time step
__global__ void calculateTimeStepKernel(double *d_rho, double *d_rhoU, double *d_rhoE, 
                                      double *d_dt_local, int startIdx, int endIdx, double dx, double CFL) {
    __shared__ double s_maxSpeed[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    s_maxSpeed[tid] = 0.0;
    
    if (i >= startIdx && i < endIdx) {
        double rho = d_rho[i];
        double u = d_rhoU[i] / rho;
        double e = d_rhoE[i] / rho;
        double p = (rho * e - 0.5 * rho * u * u) * (GAMMA - 1.0);
        double c = sqrt(GAMMA * p / rho);
        
        // Local maximum wave speed
        s_maxSpeed[tid] = fabs(u) + c;
    }
    
    __syncthreads();
    
    // Reduction to find maximum wave speed
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_maxSpeed[tid] = fmax(s_maxSpeed[tid], s_maxSpeed[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        d_dt_local[blockIdx.x] = s_maxSpeed[0];
    }
}

// Function to finalize time step calculation on CPU
double finalizeTimeStep(double *h_max_speeds, int numBlocks, double dx, double CFL) {
    double max_speed = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        max_speed = fmax(max_speed, h_max_speeds[i]);
    }
    
    // CFL condition
    if (max_speed > 0.0) {
        return CFL * dx / max_speed;
    } else {
        return 0.0001; // Fallback value
    }
}

// Function to handle multi-GPU initialization with improved work distribution
void initializeGPUs(GPUSection *sections, int numGPUs, int imax) {
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    
    if (numDevices < numGPUs) {
        printf("Requested %d GPUs but only %d available. Using %d GPUs.\n", 
               numGPUs, numDevices, numDevices);
        numGPUs = numDevices;
    }
    
    // Enable peer access between all GPUs
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        
        // Set device properties for better performance
        CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
        
        for (int j = 0; j < numGPUs; j++) {
            if (i != j) {
                int canAccessPeer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
                if (canAccessPeer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
                    printf("GPU %d can access GPU %d directly.\n", i, j);
                } else {
                    printf("GPU %d cannot access GPU %d directly.\n", i, j);
                }
            }
        }
    }
    
    // Compute grid sections for each GPU with better load balancing
// Standardized Output for Comparison
void write_standardized_solution(const char *filename, double *h_solution, int imax) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < imax; i++) {
        outfile << i << " " << h_solution[i] << "\n"; // Ensure index + value format
    }
    outfile.close();
}

// Call this function before exiting main()
write_standardized_solution("Standardized_Solution.dat", Qconv, imax);

// Standardized Output for Comparison
void write_standardized_solution(const char *filename, double *h_solution, int imax) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    outfile << "# Solution from Claude3\n";  // Include file identifier
    for (int i = 0; i < imax; i++) {
        outfile << i << " " << h_solution[i] << "\n"; // Ensure index + value format
    }
    outfile.close();
}

// Call this function before exiting main()
write_standardized_solution("Standardized_Solution_Claude3.dat", Qconv, imax);
