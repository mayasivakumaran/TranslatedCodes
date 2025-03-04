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
    cudaStream_t stream;
} GPUSection;

// Device constants
__constant__ double d_gamma;
__constant__ double d_dx;
__constant__ double d_dt;

// CUDA kernel for Roe solver interfaces
__global__ void interfacesKernel(double *d_Q, double *d_Favg, double *d_Diss, int startIdx, int endIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    if (i >= startIdx && i < endIdx) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Defining Roe Variables from Q
        RhoL = d_Q[i * 3];
        RhoR = d_Q[(i+1) * 3];
        UL = d_Q[i * 3 + 1] / RhoL;
        UR = d_Q[(i+1) * 3 + 1] / RhoR;
        EL = d_Q[i * 3 + 2] / RhoL;
        ER = d_Q[(i+1) * 3 + 2] / RhoR;
        
        PL = (RhoL * EL - 0.5 * RhoL * (UL * UL)) * (GAMMA - 1.0);
        PR = (RhoR * ER - 0.5 * RhoR * (UR * UR)) * (GAMMA - 1.0);
        HL = EL + PL / RhoL;
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg[i * 3] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg[i * 3 + 1] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg[i * 3 + 2] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
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
        for (int j = 0; j < 3; j++) {
            d_Diss[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                d_Diss[i * 3 + j] += fabs(Lambda[k]) * Alpha[k] * Eig[k][j];
            }
        }
    }
}

// CUDA kernel for supersonic inflow boundary condition
__global__ void superSonicInflowBCKernel(double *d_Q, double *d_Favg, double *d_Diss, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Defining Roe Variables
        RhoL = 1.0;
        RhoR = d_Q[(idx+1) * 3];
        UL = 2.95;
        UR = d_Q[(idx+1) * 3 + 1] / RhoR;
        EL = (1.0/GAMMA)/(GAMMA-1) + 0.5 * (RhoL * UL * UL);
        ER = d_Q[(idx+1) * 3 + 2] / RhoR;
        PL = 1.0/GAMMA;
        PR = (RhoR * ER - 0.5 * RhoR * (UR * UR)) * (GAMMA - 1.0);
        HL = EL + PL / RhoL;
        HR = ER + PR / RhoR;
        
        // Average fluxes
        d_Favg[idx * 3] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg[idx * 3 + 1] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg[idx * 3 + 2] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
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
        for (int j = 0; j < 3; j++) {
            d_Diss[idx * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                d_Diss[idx * 3 + j] += fabs(Lambda[k]) * Alpha[k] * Eig[k][j];
            }
        }
    }
}

// CUDA kernel for subsonic outflow boundary condition
__global__ void subSonicOutflowBCKernel(double *d_Q, double *d_Favg, double *d_Diss, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Variables
        double UL, UR, PL, PR, HL, HR, RhoL, RhoR, EL, ER;
        double Rhoroe, Croe, Uroe, Hroe;
        double Ci, Ti, Mi, Pstag, Mset, Pback;
        double Lambda[3], Alpha[3];
        double Eig[3][3];
        
        // Get left state variables
        RhoL = d_Q[idx * 3];
        UL = d_Q[idx * 3 + 1] / RhoL;
        UR = UL; // Same velocity at the boundary
        EL = d_Q[idx * 3 + 2] / RhoL;
        
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
        d_Favg[idx * 3] = 0.5 * (RhoR * UR + RhoL * UL);
        d_Favg[idx * 3 + 1] = 0.5 * (RhoR * (UR * UR) + PR + RhoL * (UL * UL) + PL);
        d_Favg[idx * 3 + 2] = 0.5 * (RhoR * UR * HR + RhoL * UL * HL);
        
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
        for (int j = 0; j < 3; j++) {
            d_Diss[idx * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                d_Diss[idx * 3 + j] += fabs(Lambda[k]) * Alpha[k] * Eig[k][j];
            }
        }
    }
}

// CUDA kernel for updating solution
__global__ void updateSolutionKernel(double *d_Q, double *d_Qnew, double *d_Favg, double *d_Diss, 
                                    int startIdx, int endIdx, double dt, double dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + startIdx;
    
    if (i >= startIdx && i < endIdx) {
        for (int j = 0; j < 3; j++) {
            d_Qnew[i * 3 + j] = d_Q[i * 3 + j] - (dt / dx) * 
                               ((d_Favg[i * 3 + j] - 0.5 * d_Diss[i * 3 + j]) - 
                                (d_Favg[(i-1) * 3 + j] - 0.5 * d_Diss[(i-1) * 3 + j]));
        }
    }
}

// Function to handle multi-GPU initialization
void initializeGPUs(GPUSection *sections, int numGPUs, int imax) {
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    
    if (numDevices < numGPUs) {
        printf("Requested %d GPUs but only %d available. Using %d GPUs.\n", 
               numGPUs, numDevices, numDevices);
        numGPUs = numDevices;
    }
    
    // Compute grid sections for each GPU
    int baseSize = imax / numGPUs;
    int remainder = imax % numGPUs;
    
    int startIdx = 1; // Skip first cell (boundary condition)
    
    for (int i = 0; i < numGPUs; i++) {
        sections[i].device = i;
        sections[i].startIdx = startIdx;
        
        // Distribute remainder to first few GPUs
        int sectionSize = baseSize + (i < remainder ? 1 : 0);
        
        // Account for the boundary cells
        if (i == 0) {
            // First GPU includes left boundary
            sectionSize += 0;
        }
        
        if (i == numGPUs - 1) {
            // Last GPU includes right boundary
            sectionSize += 0;
        }
        
        sections[i].endIdx = startIdx + sectionSize;
        sections[i].localSize = sectionSize;
        
        // Set up device and stream
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaStreamCreate(&sections[i].stream));
        
        startIdx = sections[i].endIdx;
    }
}

// Function to allocate memory on multiple GPUs
void allocateGPUMemory(double **d_Q, double **d_Qnew, double **d_Favg, double **d_Diss, 
                      GPUSection *sections, int numGPUs, int imax) {
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        
        // Allocate memory on this device
        CUDA_CHECK(cudaMalloc(&d_Q[i], 3 * imax * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Qnew[i], 3 * imax * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Favg[i], 3 * imax * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Diss[i], 3 * imax * sizeof(double)));
        
        // Set device constants
        double gamma = GAMMA;
        double dx = 1.0 / (double)imax;
        double dt = 0.0001;
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_gamma, &gamma, sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_dx, &dx, sizeof(double)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_dt, &dt, sizeof(double)));
    }
}

// Function to copy initial data to GPU memory
void copyInitialDataToGPUs(double *h_Q, double **d_Q, GPUSection *sections, int numGPUs, int imax) {
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaMemcpyAsync(d_Q[i], h_Q, 3 * imax * sizeof(double), 
                                  cudaMemcpyHostToDevice, sections[i].stream));
    }
    
    // Synchronize all devices
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaStreamSynchronize(sections[i].stream));
    }
}

// Function to synchronize data between GPUs
void synchronizeGPUData(double **d_Q, GPUSection *sections, int numGPUs, int imax) {
    // Temporary buffer for peer-to-peer transfers
    double *h_buffer = (double *)malloc(3 * imax * sizeof(double));
    
    // Copy data from each GPU to host
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaMemcpyAsync(&h_buffer[3 * sections[i].startIdx], 
                                  &d_Q[i][3 * sections[i].startIdx],
                                  3 * sections[i].localSize * sizeof(double),
                                  cudaMemcpyDeviceToHost, 
                                  sections[i].stream));
    }
    
    // Synchronize all streams
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaStreamSynchronize(sections[i].stream));
    }
    
    // Copy full dataset back to each GPU
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaMemcpyAsync(d_Q[i], h_buffer, 3 * imax * sizeof(double),
                                  cudaMemcpyHostToDevice, 
                                  sections[i].stream));
    }
    
    // Synchronize all streams
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(sections[i].device));
        CUDA_CHECK(cudaStreamSynchronize(sections[i].stream));
    }
    
    free(h_buffer);
}

// Function to write solution to file
void writeSolutionToFile(double *h_Q, int imax, int step, const char *baseFileName) {
    char filename[MAX_FILENAME];
    char stepStr[20];
    
    snprintf(stepStr, sizeof(stepStr), "%010d", step);
    snprintf(filename, sizeof(filename), "%sStep_%s.dat", baseFileName, stepStr);
    
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }
    
    for (int i = 1; i <= imax; i++) {
        fprintf(file, "%e %e %e\n", h_Q[i*3], h_Q[i*3+1], h_Q[i*3+2]);
    }
    
    fclose(file);
}

int main() {
    // Problem parameters
    int imax = 1000;
    double L = 1.0;
    double xmin = 0.0;
    double xmax = 1.0;
    double dx = (xmax - xmin) / imax;
    double dt = 0.0001;
    double gamma = 1.4;
    
    // Flow properties
    double Rhoin = 1.0;
    double Machin = 2.95;
    double Cin = 1.0;
    double Uin = 2.95;
    double Pin = 1.0;
    double Rhoout = 3.8106;
    double Machout = 0.4782;
    double Cout = 1.62;
    double Uout = Machout * Cout;
    double Pout = 9.9862;
    
    // Allocate host memory
    double *h_Q = (double *)malloc(3 * (imax + 1) * sizeof(double));
    double *h_Qnew = (double *)malloc(3 * (imax + 1) * sizeof(double));
    
    // Initialize solution
    // Left side (supersonic)
    for (int i = 1; i <= imax/2; i++) {
        h_Q[i*3] = Rhoin;
        h_Q[i*3+1] = Rhoin * 2.95;
        h_Q[i*3+2] = (Pin/gamma)/(gamma-1) + 0.5 * Rhoin * (Uin * Uin);
    }
    
    // Right side (subsonic)
    for (int i = imax/2 + 1; i <= imax; i++) {
        h_Q[i*3] = Rhoout;
        h_Q[i*3+1] = Rhoout * 2.95;
        h_Q[i*3+2] = (Pout/gamma)/(gamma-1) + Rhoout * 0.5 * (Uout * Uout);
    }
    
    // Copy initial values to h_Qnew
    memcpy(h_Qnew, h_Q, 3 * (imax + 1) * sizeof(double));
    
    // Write initial solution
    FILE *initialFile = fopen("./Solutions/SolutionStep_0000.dat", "w");
    for (int i = 1; i <= imax; i++) {
        fprintf(initialFile, "%e %e %e\n", h_Q[i*3], h_Q[i*3+1], h_Q[i*3+2]);
    }
    fclose(initialFile);
    
    // Set up multi-GPU configuration
    int numGPUs = 4; // Using 4 GPUs
    GPUSection *sections = (GPUSection *)malloc(numGPUs * sizeof(GPUSection));
    initializeGPUs(sections, numGPUs, imax);
    
    // Allocate device memory
    double **d_Q = (double **)malloc(numGPUs * sizeof(double *));
    double **d_Qnew = (double **)malloc(numGPUs * sizeof(double *));
    double **d_Favg = (double **)malloc(numGPUs * sizeof(double *));
    double **d_Diss = (double **)malloc(numGPUs * sizeof(double *));
    
    allocateGPUMemory(d_Q, d_Qnew, d_Favg, d_Diss, sections, numGPUs, imax + 1);
    
    // Copy initial data to all GPUs
    copyInitialDataToGPUs(h_Q, d_Q, sections, numGPUs, imax + 1);
    
    // Open file for convergence data
    FILE *convFile = fopen("Convergence.dat", "w");
    
    // Main time loop
    for (int t = 1; t <= 2000000; t++) {
        // Print progress
        if (t % 100 == 0) {
            
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
    outfile << "# Solution from Claude1\n";  // Include file identifier
    for (int i = 0; i < imax; i++) {
        outfile << i << " " << h_solution[i] << "\n"; // Ensure index + value format
    }
    outfile.close();
}

// Call this function before exiting main()
write_standardized_solution("Standardized_Solution_Claude1.dat", Qconv, imax);
