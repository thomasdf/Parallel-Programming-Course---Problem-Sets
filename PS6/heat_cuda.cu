#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

/* Functions to be implemented: */
float ftcs_solver_gpu ( int step, int block_size_x, int block_size_y );
float ftcs_solver_gpu_shared ( int step, int block_size_x, int block_size_y );
float ftcs_solver_gpu_texture ( int step, int block_size_x, int block_size_y );
void external_heat_gpu ( int step, int block_size_x, int block_size_y );
void transfer_from_gpu( int step );
void transfer_to_gpu();
void device_allocation();

/* Prototypes for functions found at the end of this file */
void write_temp( int step );
void print_local_temps();
void init_temp_material();
void init_local_temp();
void host_allocation();
void add_time(float time);
void print_time_stats();
dim3 threadsPerBlock(int block_size_x, int block_size_y);
dim3 numBlocks(int block_size_x, int block_size_y);


/*
 * Physical quantities:
 * k                    : thermal conductivity      [Watt / (meter Kelvin)]
 * rho                  : density                   [kg / meter^3]
 * cp                   : specific heat capacity    [kJ / (kg Kelvin)]
 * rho * cp             : volumetric heat capacity  [Joule / (meter^3 Kelvin)]
 * alpha = k / (rho*cp) : thermal diffusivity       [meter^2 / second]
 *
 * Mercury:
 * cp = 0.140, rho = 13506, k = 8.69
 * alpha = 8.69 / (0.140*13506) =~ 0.0619
 *
 * Copper:
 * cp = 0.385, rho = 8960, k = 401
 * alpha = 401.0 / (0.385 * 8960) =~ 0.120
 *
 * Tin:
 * cp = 0.227, k = 67, rho = 7300
 * alpha = 67.0 / (0.227 * 7300) =~ 0.040
 *
 * Aluminium:
 * cp = 0.897, rho = 2700, k = 237
 * alpha = 237 / (0.897 * 2700) =~ 0.098
 */

const float MERCURY = 0.0619;
const float COPPER = 0.116;
const float TIN = 0.040;
const float ALUMINIUM = 0.098;

/* Discretization: 5cm square cells, 2.5ms time intervals */
const float
    h  = 5e-2,
    dt = 2.5e-3;

/* Size of the computational grid - 1024x1024 square */
const int GRID_SIZE[2] = {2048, 2048};

/* Parameters of the simulation: how many steps, and when to cut off the heat */
const int NSTEPS = 10000;
const int CUTOFF = 5000;

/* How often to dump state to file (steps). */
const int SNAPSHOT = 500;

/* For time statistics */
float min_time = -2.0;
float max_time = -2.0;
float avg_time = 0.0;

/* Arrays for the simulation data, on host */
float
    *material,          // Material constants
    *temperature;       // Temperature field

/* Arrays for the simulation data, on device */
float
    *material_device,           // Material constants
    *temperature_device[2];      // Temperature field, 2 arrays 

texture<float> textureReference;



/* Allocate arrays on GPU */
void device_allocation(){
    size_t temperature_size = GRID_SIZE[0] * GRID_SIZE[1];
    size_t material_size = (GRID_SIZE[0]) * (GRID_SIZE[1]);
    cudaMalloc((void**) &temperature_device[0], sizeof(float) * temperature_size);
    cudaMalloc((void**) &temperature_device[1], sizeof(float) * temperature_size);
    cudaMalloc((void**) &material_device, sizeof(float) * material_size);
}

/* Transfer input to GPU */
void transfer_to_gpu(){
    size_t temperature_size =GRID_SIZE[0]*GRID_SIZE[1];
    size_t material_size = (GRID_SIZE[0])*(GRID_SIZE[1]);

    cudaMemcpy(temperature_device[0], temperature, sizeof(float) * temperature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(temperature_device[1], temperature, sizeof(float) * temperature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(material_device, material, sizeof(float) * material_size, cudaMemcpyHostToDevice);
}

/* Transfer output from GPU to CPU */
void transfer_from_gpu(int step){
    size_t temperature_size =GRID_SIZE[0]*GRID_SIZE[1];
    size_t material_size = (GRID_SIZE[0])*(GRID_SIZE[1]);

    cudaMemcpy(temperature, temperature_device[step%2], sizeof(float) * temperature_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(material_device, material, sizeof(float) * material_size, cudaMemcpyDeviceToHost);
}

__device__ int sharedindex(int x, int y, int sharedXsize){
    return(((y+1)*(sharedXsize)) + (x+1));
}

__device__ int globalti(int i, int j, int XSIZE, int YSIZE){
    if(i < 0){
        i++;
    }
    if(i >= XSIZE){
        i--;
    }
    if(j < 0){
        j++;
    }
    if(j >= YSIZE){
        j--;
    }
    return((j)*(XSIZE) + i);
}

/* Plain/global memory only kernel */
__global__ void  ftcs_kernel( float* in, float* out, float* material_device, int step, int XSIZE, int YSIZE ){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int arrayIndex = (j)*(XSIZE) + i;

    if(i < XSIZE && j < YSIZE){
        out[arrayIndex] = in[globalti(i, j, XSIZE, YSIZE)] + material_device[arrayIndex] *
            (
                in[globalti(i+1, j, XSIZE, YSIZE)] + 
                in[globalti(i-1, j, XSIZE, YSIZE)] +
                in[globalti(i, j+1, XSIZE, YSIZE)] +
                in[globalti(i, j-1, XSIZE, YSIZE)] -
                4*in[globalti(i, j, XSIZE, YSIZE)] 
            );
    }
}

/* Shared memory kernel */
__global__ void  ftcs_kernel_shared( float* in, float* out, float* material_device, int step, int XSIZE, int YSIZE ){
    extern __shared__ float inShared[];
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int arrayIndex = (j)*(XSIZE) + i;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int sXsize = blockDim.x + 2;
    int sYsize = blockDim.y + 2;

    if(i < XSIZE && j < YSIZE){

        inShared[sharedindex(x, y, sXsize)] = in[globalti(i, j, XSIZE, YSIZE)];
        //edges
        if(x == 0){
            inShared[sharedindex(x-1, y, sXsize)] = in[globalti(i-1, j, XSIZE, YSIZE)];
        }
        if(x == blockDim.x){
            inShared[sharedindex(x+1, y, sXsize)] = in[globalti(i+1, j, XSIZE, YSIZE)];
        }
        if(y == 0){
            inShared[sharedindex(x, y-1, sXsize)] = in[globalti(i, j-1, XSIZE, YSIZE)];
        }
        if(y == blockDim.y){
            inShared[sharedindex(x, y+1, sXsize)] = in[globalti(i, j+1, XSIZE, YSIZE)];
        }
        //corners
        if(x == 0 && y == 0){
            inShared[sharedindex(x-1, y-1, sXsize)] = in[globalti(i-1, j-1, XSIZE, YSIZE)];
        }
        if(x == blockDim.x && y == blockDim.y){
            inShared[sharedindex(x+1, y+1, sXsize)] = in[globalti(i+1, j+1, XSIZE, YSIZE)];
        }    
        if(x == 0 && y == blockDim.y){
            inShared[sharedindex(x-1, y+1, sXsize)] = in[globalti(i-1, j+1, XSIZE, YSIZE)];
        }
        if(x == blockDim.x && y == 0){
            inShared[sharedindex(x+1, y-1, sXsize)] = in[globalti(i+1, j-1, XSIZE, YSIZE)];
        }
    }
    //shared memory loaded. Sync threads:
    __syncthreads();
    //out[arrayIndex] = 0;

    if(i < XSIZE && i > 0 && j > 0 && j < YSIZE){
        out[globalti(i, j, XSIZE, YSIZE)] = inShared[sharedindex(x, y, sXsize)] + material_device[arrayIndex] *
            (
                inShared[sharedindex(x+1, y, sXsize)] +
                inShared[sharedindex(x-1, y, sXsize)] +
                inShared[sharedindex(x, y+1, sXsize)] +
                inShared[sharedindex(x, y-1, sXsize)] -
                4*inShared[sharedindex(x, y, sXsize)]
            );
        //out[arrayIndex] = inShared[sharedindex(x, y, sXsize)] / 1.1;
    }
    __syncthreads();
}

/* Texture memory kernel */
__global__ void  ftcs_kernel_texture( float* in, float* out, float* material_device, int step, int XSIZE, int YSIZE ){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int arrayIndex = (j)*(XSIZE) + i;

     if(i < XSIZE && j < YSIZE){
		// out[arrayIndex] = tex2D(textureReference, i, j) + material_device[arrayIndex] *
  //           (
  //               tex2D(textureReference, ip1, j) + 
  //               tex2D(textureReference, im1, j) +
  //               tex2D(textureReference, i, jp1) +
  //               tex2D(textureReference, i, jm1) -
  //               4*tex2D(textureReference, i, j) 
  //           );
        // out[arrayIndex] = in[globalti(i, j, XSIZE, YSIZE)] + material_device[arrayIndex] *
        //     (
        //         in[globalti(i+1, j, XSIZE, YSIZE)] + 
        //         in[globalti(i-1, j, XSIZE, YSIZE)] +
        //         in[globalti(i, j+1, XSIZE, YSIZE)] +
        //         in[globalti(i, j-1, XSIZE, YSIZE)] -
        //         4*in[globalti(i, j, XSIZE, YSIZE)] 
        //     );
        //out[arrayIndex] = 0;

		out[arrayIndex] = tex1Dfetch(textureReference, globalti(i, j, XSIZE, YSIZE)) + material_device[arrayIndex] *
            (
                tex1Dfetch(textureReference, globalti(i+1, j, XSIZE, YSIZE)) + 
                tex1Dfetch(textureReference, globalti(i-1, j, XSIZE, YSIZE)) +
                tex1Dfetch(textureReference, globalti(i, j+1, XSIZE, YSIZE)) +
                tex1Dfetch(textureReference, globalti(i, j-1, XSIZE, YSIZE)) -
                4*tex1Dfetch(textureReference, globalti(i, j, XSIZE, YSIZE)) 
            );
    }
}


/* External heat kernel, should do the same work as the external
 * heat function in the serial code 
 */
__global__ void external_heat_kernel( float* temperature_device, int step, int XSIZE, int YSIZE ){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    int arrayIndex = j*XSIZE + i;
    if(i < XSIZE && j < YSIZE){
        if( i >= (XSIZE/4) && i < (3*XSIZE/4) && j > ((YSIZE/2) - (YSIZE/16)) && j < ((YSIZE/2) + (YSIZE/16)) ){
            temperature_device[arrayIndex] = 100.0;
        }
    }
}

/* Set up and call ftcs_kernel
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu( int step, int block_size_x, int block_size_y ){
    dim3 num_blocks = numBlocks(block_size_x, block_size_y);
    dim3 threads_per_block = threadsPerBlock(block_size_x, block_size_y);

    if(step == 0){
        printf("numblocks: (%d, %d)\n", num_blocks.x, num_blocks.y);
        printf("threadsPerBlock: (%d, %d)\n", threads_per_block.x, threads_per_block.y);
    }

    clock_t start = clock(), diff;
    ftcs_kernel<<<num_blocks, threads_per_block>>>(temperature_device[step%2], temperature_device[(step+1)%2], material_device, step, GRID_SIZE[0], GRID_SIZE[1]);
    diff = clock() - start;

    float time = (float)(diff * 1000) / CLOCKS_PER_SEC;
    return time;
}

/* Set up and call ftcs_kernel_shared
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu_shared( int step, int block_size_x, int block_size_y ){
    dim3 num_blocks = numBlocks(block_size_x, block_size_y);
    dim3 threads_per_block = threadsPerBlock(block_size_x, block_size_y);

    if(step == 0){
        printf("numblocks: (%d, %d)\n", num_blocks.x, num_blocks.y);
        printf("threadsPerBlock: (%d, %d)\n", threads_per_block.x, threads_per_block.y);
    }

    int sharedMemSize = (block_size_x+2)*(block_size_y+2)*sizeof(float);
    clock_t start = clock(), diff;
    ftcs_kernel_shared<<<num_blocks, threads_per_block, sharedMemSize>>>(temperature_device[step%2], temperature_device[(step+1)%2], material_device, step, GRID_SIZE[0], GRID_SIZE[1]);
    diff = clock() - start;

    float time = (float)(diff * 1000) / CLOCKS_PER_SEC;
    return time;
}

/* Set up and call ftcs_kernel_texture
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu_texture( int step, int block_size_x, int block_size_y ){
    dim3 num_blocks = numBlocks(block_size_x, block_size_y);
    dim3 threads_per_block = threadsPerBlock(block_size_x, block_size_y);

    if(step == 0){
        printf("numblocks: (%d, %d)\n", num_blocks.x, num_blocks.y);
        printf("threadsPerBlock: (%d, %d)\n", threads_per_block.x, threads_per_block.y);
    }

    //TODO: Set up texture memory
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();	

    int XSIZE = GRID_SIZE[0];
    int YSIZE = GRID_SIZE[1];
    //cudaBindTexture(NULL, textureReference, temperature_device[step%2], XSIZE, YSIZE, XSIZE*YSIZE*sizeof(float));
    cudaBindTexture(NULL, textureReference, temperature_device[step%2], XSIZE*YSIZE*sizeof(float));

    clock_t start = clock(), diff;
    ftcs_kernel_texture<<<num_blocks, threads_per_block>>>(temperature_device[step%2], temperature_device[(step+1)%2], material_device, step, GRID_SIZE[0], GRID_SIZE[1]);
    diff = clock() - start;

    float time = (float)(diff * 1000) / CLOCKS_PER_SEC;
    return time;
}


/* Set up and call external_heat_kernel */
void external_heat_gpu( int step, int block_size_x, int block_size_y ){
    dim3 num_blocks = numBlocks(block_size_x, block_size_y);
    dim3 threads_per_block = threadsPerBlock(block_size_x, block_size_y);

    external_heat_kernel<<<num_blocks, threads_per_block>>>(temperature_device[step%2], step, GRID_SIZE[0], GRID_SIZE[1]);
}

dim3 threadsPerBlock(int block_size_x, int block_size_y){
    dim3 threadsPerBlock(block_size_x, block_size_y);
    return threadsPerBlock;
}

dim3 numBlocks(int block_size_x, int block_size_y){
    int XSIZE = GRID_SIZE[0];
    int YSIZE = GRID_SIZE[1];

    //calculate number of blocks needed in X-direction
    int numblocksX = (int) ceil((float)XSIZE/(float)block_size_x);
    int numblocksY = (int) ceil((float)YSIZE/(float)block_size_y);

    dim3 numblocks(numblocksX, numblocksY);
    return numblocks;
}

void print_gpu_info(){
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  printf("Number of CUDA devices: %d\n", n_devices);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  printf("CUDA device name: %s\n" , device_prop.name);
  printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
}


int main ( int argc, char **argv ){
    
    // Parse command line arguments
    int version = 0;
    int block_size_x = 0;
    int block_size_y = 0;
    if(argc != 4){
        printf("Useage: %s <version> <block_size_x> <block_size_y>\n\n<version> can be:\n0: plain\n1: shared memory\n2: texture memory\n", argv[0]);
        exit(0);
    }
    else{
        version = atoi(argv[1]);
        block_size_x = atoi(argv[2]);
        block_size_y = atoi(argv[3]);
    }
    
    print_gpu_info();
    
    // Allocate and initialize data on host
    host_allocation();
    init_temp_material();
    
    // Allocate arrays on device, and transfer inputs
    device_allocation();
    transfer_to_gpu();
        
    // Main integration loop
    for( int step=0; step<NSTEPS; step += 1 ){
        
        if( step < CUTOFF ){
            external_heat_gpu ( step, block_size_x, block_size_y );
        }
        
        float time;
        // Call selected version of ftcs slover
        if(version == 2){
            time = ftcs_solver_gpu_texture( step, block_size_x, block_size_y );
        }
        else if(version == 1){
            time = ftcs_solver_gpu_shared(step, block_size_x, block_size_y);
        }
        else{
            time = ftcs_solver_gpu(step, block_size_x, block_size_y);
        }
        
        add_time(time);
        
        if((step % SNAPSHOT) == 0){
            // Transfer output from device, and write to file
            transfer_from_gpu(step);
            write_temp(step);
        }
    }
    
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    print_time_stats();
        
    exit ( EXIT_SUCCESS );
}


void host_allocation(){
    size_t temperature_size =GRID_SIZE[0]*GRID_SIZE[1];
    temperature = (float*) calloc(temperature_size, sizeof(float));
    size_t material_size = (GRID_SIZE[0])*(GRID_SIZE[1]); 
    material = (float*) calloc(material_size, sizeof(float));
}


void init_temp_material(){
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[y * GRID_SIZE[0] + x] = 10.0;

        }
    }
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[y * GRID_SIZE[0] + x] = 20.0;
            material[y * GRID_SIZE[0] + x] = MERCURY * (dt/(h*h));
        }
    }
    
    /* Set up the two blocks of copper and tin */
    for(int x=(5*GRID_SIZE[0]/8); x<(7*GRID_SIZE[0]/8); x++ ){
        for(int y=(GRID_SIZE[1]/8); y<(3*GRID_SIZE[1]/8); y++ ){
            material[y * GRID_SIZE[0] + x] = COPPER * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 60.0;
        }
    }
    
    for(int x=(GRID_SIZE[0]/8); x<(GRID_SIZE[0]/2)-(GRID_SIZE[0]/8); x++ ){
        for(int y=(5*GRID_SIZE[1]/8); y<(7*GRID_SIZE[1]/8); y++ ){
            material[y * GRID_SIZE[0] + x] = TIN * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 60.0;
        }
    }

    /* Set up the heating element in the middle */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            material[y * GRID_SIZE[0] + x] = ALUMINIUM * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 100.0;
        }
    }
}


void add_time(float time){
    avg_time += time;
    
    if(time < min_time || min_time < -1.0){
        min_time = time;
    }
    
    if(time > max_time){
        max_time = time;
    }
}

void print_time_stats(){
    printf("Kernel execution time (min, max, avg): %f %f %f\n", min_time, max_time, avg_time/NSTEPS);
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down
 * Only works for images which dimensions are powers of two
 */
void savebmp(char *name, unsigned char *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  unsigned char header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, GRID_SIZE[0] * GRID_SIZE[1] * 3, f);
  fclose(f);
}

void fancycolour(unsigned char *p, float temp) {
    
    if(temp <= 25){
        p[2] = 0;
        p[1] = (unsigned char)((temp/25)*255);
        p[0] = 255;
    }
    else if (temp <= 50){
        p[2] = 0;
        p[1] = 255;
        p[0] = 255 - (unsigned char)(((temp-25)/25) * 255);
    }
    else if (temp <= 75){
        
        p[2] = (unsigned char)(255* (temp-50)/25);
        p[1] = 255;
        p[0] = 0;
    }
    else{
        p[2] = 255;
        p[1] = 255 -(unsigned char)(255* (temp-75)/25) ;
        p[0] = 0;
    }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(char* filename){
    unsigned char *buffer = (unsigned char*)calloc(GRID_SIZE[0] * GRID_SIZE[1]* 3, 1);
    for (int j = 0; j < GRID_SIZE[1]; j++) {
        for (int i = 0; i < GRID_SIZE[0]; i++) {
        int p = ((GRID_SIZE[1] - j - 1) * GRID_SIZE[0] + i) * 3;
        fancycolour(buffer + p, temperature[j*GRID_SIZE[0] + i]);
      }
    }
    /* write image to disk */
    savebmp(filename, buffer, GRID_SIZE[0], GRID_SIZE[1]);
    free(buffer);
}


void write_temp (int step ){
    char filename[15];
    sprintf ( filename, "data/%.4d.bmp", step/SNAPSHOT );

    output ( filename );
    printf ( "Snapshot at step %d\n", step );
}
