#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

// Constants
#define BLOCK_SIZE 256
#define GAMMA 1.4
#define RD double
#define MAX_FILENAME 256
#define HALO_SIZE 1  // Number of halo cells for each section

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
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
    cudaStream_t transferStream;
    cudaEvent_t computeDone;
} GPUSection;

// Structure for solution variables (SoA layout for better memory access)
typedef struct {
    double *rho;     // Density
    double *rhoU;    // Momentum
    double *rhoE;    // Energy
} SolutionVariables;

// Device constants
__constant__ double d_gamma;
__constant__ double d_dx;
__constant__ double d_dt;

// CUDA kernel for Roe solver interfaces with improved memory layout
__global__ void interfacesKernel(double *d_rho, double *d_rhoU, double *d_rhoE,
                              double *d_Favg_rho, double *d_Favg_rhoU, double *d_Favg_rhoE,
                              double *d_Diss_rho, double *d_Diss_rhoU, double *d_Diss_rhoE,
                              int startIdx, int endIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    if (i >= startIdx && i < endIdx) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Defining Roe Variables from Q
        RhoL = d_rho[i];
        RhoR = d_rho[i+1];
        UL = d_rhoU[i] / RhoL;
        UR = d_rhoU[i+1] / RhoR;
        EL = d_rhoE[i] / RhoL;
        ER = d_rhoE[i+1] / RhoR;
        
        PL = (RhoL * EL - 0.5 * RhoL * (UL * UL)) * (GAMMA - 1.0);
        PR = (RhoR * ER - 0.5 * RhoR * (UR * UR)) * (GAMMA - 1.0);
        HL = EL + PL / RhoL;
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg_rho[i] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg_rhoU[i] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg_rhoE[i] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
        // Roe Variables At Interface
        Rhoroe = sqrt(RhoL * RhoR);
        Uroe = (sqrt(RhoL) * UL + sqrt(RhoR) * UR) / (sqrt(RhoL) + sqrt(RhoR));
        Hroe = (sqrt(RhoL) * HL + sqrt(RhoR) * HR) / (sqrt(RhoL) + sqrt(RhoR));
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
        
        // Alpha coefficients
        Alpha[0] = (0.25 * Uroe / Croe) * (RhoR - RhoL) * (2.0 + (GAMMA - 1.0) * Uroe / Croe) -
                   (0.5 / Croe) * (1.0 + (GAMMA - 1.0) * Uroe / Croe) * (RhoR * UR - RhoL * UL) +
                   0.5 * (GAMMA - 1.0) * (RhoR * ER - RhoL * EL) / (Croe * Croe);
        
        Alpha[1] = (1.0 - 0.5 * ((Uroe / Croe) * (Uroe / Croe)) * (GAMMA - 1.0)) * (RhoR - RhoL) +
                   (GAMMA - 1.0) * Uroe * (RhoR * UR - RhoL * UL) / (Croe * Croe) -
                   (GAMMA - 1.0) / (Croe * Croe) * (RhoR * ER - RhoL * EL);
        
        Alpha[2] = -(0.25 * Uroe / Croe) * (2.0 - (GAMMA - 1.0) * Uroe / Croe) * (RhoR - RhoL) +
                   (0.5 / Croe) * (1.0 - (GAMMA - 1.0) * Uroe / Croe) * (RhoR * UR - RhoL * UL) +
                   0.5 * (GAMMA - 1.0) * (RhoR * ER - RhoL * EL) / (Croe * Croe);
        
        // Calculate dissipation - using improved memory layout
        d_Diss_rho[i] = 0.0;
        d_Diss_rhoU[i] = 0.0;
        d_Diss_rhoE[i] = 0.0;
        
        for (int k = 0; k < 3; k++) {
            d_Diss_rho[i] += fabs(Lambda[k]) * Alpha[k] * Eig[k][0];
            d_Diss_rhoU[i] += fabs(Lambda[k]) * Alpha[k] * Eig[k][1];
            d_Diss_rhoE[i] += fabs(Lambda[k]) * Alpha[k] * Eig[k][2];
        }
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
        
        // Defining Roe Variables
        RhoL = 1.0;
        RhoR = d_rho[idx+1];
        UL = 2.95;
        UR = d_rhoU[idx+1] / RhoR;
        EL = (1.0/GAMMA)/(GAMMA-1) + 0.5 * (UL * UL);
        ER = d_rhoE[idx+1] / RhoR;
        PL = 1.0/GAMMA;
        PR = (RhoR * ER - 0.5 * RhoR * (UR * UR)) * (GAMMA - 1.0);
        HL = EL + PL / RhoL;
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg_rho[idx] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg_rhoU[idx] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg_rhoE[idx] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
        // Roe Variables At Interface
        Rhoroe = sqrt(RhoL * RhoR);
        Uroe = (sqrt(RhoL) * UL + sqrt(RhoR) * UR) / (sqrt(RhoL) + sqrt(RhoR));
        Hroe = (sqrt(RhoL) * HL + sqrt(RhoR) * HR) / (sqrt(RhoL) + sqrt(RhoR));
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
        
        // Alpha coefficients
        Alpha[0] = (0.25 * Uroe / Croe) * (RhoR - RhoL) * (2.0 + (GAMMA - 1.0) * Uroe / Croe) -
                   (0.5 / Croe) * (1.0 + (GAMMA - 1.0) * Uroe / Croe) * (RhoR * UR - RhoL * UL) +
                   0.5 * (GAMMA - 1.0) * (RhoR * ER - RhoL * EL) / (Croe * Croe);
        
        Alpha[1] = (1.0 - 0.5 * ((Uroe / Croe) * (Uroe / Croe)) * (GAMMA - 1.0)) * (RhoR - RhoL) +
                   (GAMMA - 1.0) * Uroe * (RhoR * UR - RhoL * UL) / (Croe * Croe) -
                   (GAMMA - 1.0) / (Croe * Croe) * (RhoR * ER - RhoL * EL);
        
        Alpha[2] = -(0.25 * Uroe / Croe) * (2.0 - (GAMMA - 1.0) * Uroe / Croe) * (RhoR - RhoL) +
                   (0.5 / Croe) * (1.0 - (GAMMA - 1.0) * Uroe / Croe) * (RhoR * UR - RhoL * UL) +
                   0.5 * (GAMMA - 1.0) * (RhoR * ER - RhoL * EL) / (Croe * Croe);
        
        // Calculate dissipation
        d_Diss_rho[idx] = 0.0;
        d_Diss_rhoU[idx] = 0.0;
        d_Diss_rhoE[idx] = 0.0;
        
        for (int k = 0; k < 3; k++) {
            d_Diss_rho[idx] += fabs(Lambda[k]) * Alpha[k] * Eig[k][0];
            d_Diss_rhoU[idx] += fabs(Lambda[k]) * Alpha[k] * Eig[k][1];
            d_Diss_rhoE[idx] += fabs(Lambda[k]) * Alpha[k] * Eig[k][2];
        }
    }
}

// CUDA kernel for subsonic outflow boundary condition
__global__ void subSonicOutflowBCKernel(double *d_rho, double *d_rhoU, double *d_rhoE,
                                      double *d_Favg_rho, double *d_Favg_rhoU, double *d_Favg_rhoE,
                                      double *d_Diss_rho, double *d_Diss_rhoU, double *d_Diss_rhoE,
                                      int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Ci, Ti, Mi, Pstag, Mset, Pback;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Get left state variables
        RhoL = d_rho[idx];
        UL = d_rhoU[idx] / RhoL;
        UR = UL; // Same velocity at the boundary
        EL = d_rhoE[idx] / RhoL;
        
        // Calculate pressure and other variables
        PL = (RhoL * EL - 0.5 * RhoL * (UL * UL)) * (GAMMA - 1.0);
        
        Ti = GAMMA * PL / RhoL;
        Ci = sqrt(Ti);
        Mi = UL / Ci;
        Pstag = PL * pow((1.0 + 0.5 * (GAMMA - 1.0) * Mi * Mi), (GAMMA / (GAMMA - 1.0)));
        Mset = 0.4782;
        Pback = Pstag * pow((1.0 + 0.5 * (GAMMA - 1.0) * Mset * Mset), (-GAMMA / (GAMMA - 1.0)));
        RhoR = (Pback * GAMMA) / Ti;
        
        PR = Pback;
        
        // These values are undefined in original code, setting to reasonable values
        ER = PR / ((GAMMA - 1.0) * RhoR) + 0.5 * UR * UR;
        
        HL = EL + PL / RhoL;
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg_rho[idx] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg_rhoU[idx] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg_rhoE[idx] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
        // Roe Variables At Interface
        Rhoroe = sqrt(RhoL * RhoR);
        Uroe = (sqrt(RhoL) * UL + sqrt(RhoR) * UR) / (sqrt(RhoL) + sqrt(RhoR));
        Hroe = (sqrt(RhoL) * HL + sqrt(RhoR) * HR) / (sqrt(RhoL) + sqrt(RhoR));
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
        
        // Alpha coefficients
        Alpha[0] = (0.25 * Uroe / Croe) * (RhoR - RhoL) * (2.0 + (GAMMA - 1.0) * Uroe / Croe) -
                   (0.5 / Croe) * (1.0 + (GAMMA - 1.0) * Uroe / Croe) * (RhoR * UR - RhoL * UL) +
                   0.5 * (GAMMA - 1.0) * (RhoR * ER - RhoL * EL) / (Croe * Croe);
        
        Alpha[1] = (1.0 - 0.5 * ((Uroe / Croe) * (Uroe / Croe)) * (GAMMA - 1.0)) * (RhoR - RhoL) +
                   (GAMMA - 1.0) * Uroe * (RhoR * UR - RhoL * UL) / (Croe * Croe) -
                   (GAMMA - 1.0) / (Croe * Croe) * (RhoR * ER - RhoL * EL);
        
        Alpha[2] = -(0.25 * Uroe / Croe) * (2.0 - (GAMMA - 1.0) * Uroe / Croe) * (RhoR - RhoL) +
                   (0.5 / Croe) * (1.0 - (GAMMA - 1.0) * Uroe / Croe) * (RhoR * UR - RhoL * UL) +
                   0.5 * (GAMMA - 1.0) * (RhoR * ER - RhoL * EL) / (Croe * Croe);
        
        // Calculate dissipation
        d_Diss_rho[idx] = 0.0;
        d_Diss_rhoU[idx] = 0.0;
        d_Diss_rhoE[idx] = 0.0;
        
        for (int k = 0; k < 3; k++) {
            d_Diss_rho[idx] += fabs(Lambda[k]) * Alpha[k] * Eig[k][0];
            d_Diss_rhoU[idx] += fabs(Lambda[k]) * Alpha[k] * Eig[k][1];
            d_Diss_rhoE[idx] += fabs(Lambda[k]) * Alpha[k] * Eig[k][2];
        }
    }
}

// CUDA kernel for updating solution
__global__ void updateSolutionKernel(double *d_rho, double *d_rhoU, double *d_rhoE,
                                    double *d_rho_new, double *d_rhoU_new, double *d_rhoE_new,
                                    double *d_Favg_rho, double *d_Favg_rhoU, double *d_Favg_rhoE,
                                    double *d_Diss_rho, double *d_Diss_rhoU, double *d_Diss_rhoE,
                                    int startIdx, int endIdx, double dt, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    if (i >= startIdx && i < endIdx) {
        // Update density
        d_rho_new[i] = d_rho[i] - (dt / dx) * 
                       ((d_Favg_rho[i] - 0.5 * d_Diss_rho[i]) - 
                        (d_Favg_rho[i-1] - 0.5 * d_Diss_rho[i-1]));
        
        // Update momentum
        d_rhoU_new[i] = d_rhoU[i] - (dt / dx) * 
                        ((d_Favg_rhoU[i] - 0.5 * d_Diss_rhoU[i]) - 
                         (d_Favg_rhoU[i-1] - 0.5 * d_Diss_rhoU[i-1]));
        
        // Update energy
        d_rhoE_new[i] = d_rhoE[i] - (dt / dx) * 
                        ((d_Favg_rhoE[i] - 0.5 * d_Diss_rhoE[i]) - 
                         (d_Favg_rhoE[i-1] - 0.5 * d_Diss_rhoE[i-1]));
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
    int baseSize = imax / numGPUs;
    int remainder = imax % numGPUs;
    
    int startIdx = 1; // Skip first cell (boundary condition)
    
    for (int i = 0; i < numGPUs; i++) {
        sections[i].device = i;
        sections[i].startIdx = startIdx;
        
        // Distribute remainder to first few GPUs
        int sectionSize = baseSize + (i < remainder ? 1 : 0);
        
        sections[i].endIdx = startIdx + sectionSize;
        sections[i].localSize = sectionSize;
        
        // Set halo regions
        if (i > 0) {
            sections[i].haloStartL = sections[i].startIdx - HALO_SIZE;
            sections[i].haloEndL = sections[i].startIdx;
        } else {
            // Special case for first GPU - no left halo needed
            sections[i].haloStartL = 0;
            sections[i].haloEndL = 0;
        }
        
        if (i < numGPUs - 1) {
            sections[i].haloStartR = sections[i].endIdx;
            sections[i].haloEndR = sections[i].endIdx + HALO_SIZE;
        } else {
            // Special case for last GPU - no right halo needed
            sections[i].haloStartR = 0;
            sections[i].haloEndR = 0;
        }
        
        // Set up device, streams, and events
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaStreamCreate(&sections[i].computeStream));
        CUDA_CHECK(cudaStreamCreate(&sections[i].transferStream));
        CUDA_CHECK(cudaEventCreate(&sections[i].computeDone));
        
        startIdx = sections[i].endIdx;
        
        printf("GPU %d: range [%d,%d), local size %d, left halo [%d,%d), right halo [%d,%d)\n",
               i, sections[i].startIdx, sections[i].endIdx, sections[i].localSize,
               sections[i].haloStartL, sections[i].haloEndL,
               sections[i].haloStartR, sections[i].haloEndR);
    }
}

// Allocate SoA memory layout for better memory coalescing
void allocateSolutionMemory(SolutionVariables *vars, int size) {
    CUDA_CHECK(cudaMalloc(&vars->rho, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vars->rhoU, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vars->rhoE, size * sizeof(double)));
}

// Free memory for SoA layout
void freeSolutionMemory(SolutionVariables *vars) {
    CUDA_CHECK(cudaFree(vars->rho));
    CUDA_CHECK(cudaFree(vars->rhoU));
    CUDA_CHECK(cudaFree(vars->rhoE));
}

// Function to allocate memory on multiple GPUs with improved layout
void allocateGPUMemory(SolutionVariables **d_Q, SolutionVariables **d_Qnew,
                      SolutionVariables **d_Favg, SolutionVariables **d_Diss,
                      GPUSection *sections, int numGPUs, int imax) {
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        
        // Allocate memory on this device
        d_Q[i] = (SolutionVariables *)malloc(sizeof(SolutionVariables));
        d_Qnew[i] = (SolutionVariables *)malloc(sizeof(SolutionVariables));
        d_Favg[i] = (SolutionVariables *)malloc(sizeof(SolutionVariables));
        d_Diss[i] = (SolutionVariables *)malloc(sizeof(SolutionVariables));
        
        allocateSolutionMemory(d_Q[i], imax + 1);
        allocateSolutionMemory(d_Qnew[i], imax + 1);
        allocateSolutionMemory(d_Favg[i], imax + 1);
        allocateSolutionMemory(d_Diss[i], imax + 1);
        
        // Set device constants
        double gamma = GAMMA;
        double dx = 1.0 / (double)imax;
        double dt = 0.0001;
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_gamma, &gamma, sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_dx, &dx, sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_dt, &dt, sizeof(double)));
    }
}

// Function to copy initial data to GPU memory with SoA layout
void copyInitialDataToGPUs(double *h_rho, double *h_rhoU, double *h_rhoE,
                         SolutionVariables **d_Q, GPUSection *sections, int numGPUs, int imax) {
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        
        // Copy each variable array separately
        CUDA_CHECK(cudaMemcpyAsync(d_Q[i]->rho, h_rho, imax * sizeof(double),
                                  cu
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
    outfile << "# Solution from Claude2\n";  // Include file identifier
    for (int i = 0; i < imax; i++) {
        outfile << i << " " << h_solution[i] << "\n"; // Ensure index + value format
    }
    outfile.close();
}

// Call this function before exiting main()
write_standardized_solution("Standardized_Solution_Claude2.dat", Qconv, imax);
