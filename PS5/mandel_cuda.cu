#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

/* Shorthand for less typing */
typedef unsigned char uchar;

/* Declarations of output functions */
void output(int* pixel);
void fancycolour(uchar *p, int iter);
void savebmp(char *name, uchar *buffer, int x, int y);

/* Struct for complex numbers */
typedef struct {
  float real, imag;
} complex_t;

/* Size of image, in pixels */
const int XSIZE = 8000;
const int YSIZE = 8000;

/* Max number of iterations */
const int MAXITER = 1024;

/* Range in x direction */
const float xleft = -2.0;
const float xright = 1.0;
const float ycenter = 0.0;

/* Range in y direction, calculated in main
 * based on range in x direction and image size
 */
float yupper, ylower;

/* Distance between numbers */
float step;


/* Timing */
double walltime() {
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec + 1e-6 * t.tv_usec);
}

// __global__ complex_t createComplex(int index, int xsize, int xleft, int ylower, int step){
//   complex_t c;
//   int* coordinatePair = toIJindex(index, xsize);
//   int i = coordinatePair[0];
//   int j = coordinatePair[1];
//   c.real = (xleft + step * i);
//   c.imag = (ylower + step * j);
//   return c;
// }

// __global__ int toOneDIndex(int i, int j, xsize){
//   return i + (j * xsize);
// }

// __global__ int* toIJindex(index, xsize){
//   int* coordinatePair = malloc((void**)&coordinatePair, sizeof(int)*2);
//   coordinatePair[0] = index % xsize;
//   coordinatePair[1] = floor(index / xsize);
//   return coordinatePair;
// }

/* Acutal GPU kernel which will be executed in parallel on the GPU */
__global__ void mandel_kernel( int* gpu_pixel_array, int blocksize, int numblocks, float ylower, float yupper, float xleft, float step, int xsize ){

  int currentIndex = blockIdx.x * blockDim.x + threadIdx.x;

  float real, imag, tempreal, tempimag, tempreal2, tempimag2;
  // get 2D index from 1D-index
  int i,j;
  
  j = (int) (currentIndex / xsize);
  i = currentIndex % xsize;
  
  //init c to correct value based on index
  real = (xleft + step * i);
  imag = (ylower + step * j);


      int iter = 0;
      tempreal = real;
      tempimag = imag;
      
      //if(currentIndex == 1311360){ 3277824
      // if(currentIndex == 2622720){
      //   printf("step = %f\n", step);
      //   printf("i = %d\n", i);
      //   printf("j = %d\n", j);
      //   printf("step * i = %f\n", step*i);
      //   printf("step * j = %f\n", step*j);
      //   printf("real: %f\n", real);
      //   printf("imag: %f\n", imag);
      // }

      while (tempreal * tempreal + tempimag * tempimag < 4) {
        tempreal2 = tempreal * tempreal - tempimag * tempimag + real;
        tempimag2 = 2 * tempreal * tempimag + imag;
        tempreal = tempreal2;
        tempimag = tempimag2;
        iter++;
        if(iter == MAXITER){
            break;
        }
      }

      // if(currentIndex == 2622720){
      //   printf("iter = %d\n", iter);
      //   printf("step = %f\n", step);
      // }

  // while( (x*x + y*y) < 4 && count < MAXITER){
  //     count++;
  //     zreal = x*x - y*y + x;
  //     y = 2 * x * x + y;
  //     x = zreal;
  // }

  //place at correct index in row-major order
      // if(currentIndex == 2622720){
      //   printf("i + (j * xsize) = %d\n", i + (j * xsize));
      // }
  currentIndex = i + (j * xsize);
  gpu_pixel_array[currentIndex] = iter;
}

/* Set up and call GPU kernel */
void calculate_cuda(int* pixel){
    // Allocate memory
  int* gpu_pixel_array;
  cudaMalloc((void**) &gpu_pixel_array, sizeof(int)*XSIZE*YSIZE);
    // Compute thread-block size
  int number_of_pixels = XSIZE * YSIZE;
  int block_size = 1024;
  int number_of_threadblocks = number_of_pixels/block_size;
  printf("number of threadblocks = %d\n", number_of_threadblocks);
  printf("number of threads per block = %d\n", block_size );
  printf("threads total = %d, pixels total = %d\n", number_of_threadblocks*block_size, number_of_pixels );
    // Call kernel
  mandel_kernel<<<number_of_threadblocks, block_size>>>(gpu_pixel_array, block_size, number_of_threadblocks, ylower, yupper, xleft, step, XSIZE);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()) );
    // Transfer result from GPU to CPU
  cudaMemcpy(pixel, gpu_pixel_array, sizeof(int)*XSIZE*YSIZE, cudaMemcpyDeviceToHost);
}
    

/* Calculate the number of iterations until divergence for each pixel.
 * If divergence never happens, return MAXITER
 */
void calculate(int* pixel) {
  for (int i = 0; i < XSIZE; i++) {
    for (int j = 0; j < YSIZE; j++) {
      complex_t c, z, temp;
      int iter = 0;
      c.real = (xleft + step * i);
      c.imag = (ylower + step * j);
      z = c;
      while (z.real * z.real + z.imag * z.imag < 4) {
        temp.real = z.real * z.real - z.imag * z.imag + c.real;
        temp.imag = 2 * z.real * z.imag + c.imag;
        z = temp;
        iter++;
        if(iter == MAXITER){
            break;
        }
      }
      pixel[j * XSIZE + i] = iter;
    }
  }
}


int main(int argc, char **argv) {
    
    /* Check input arguments */
  if (argc == 1) {
    puts("Usage: MANDEL n");
    puts("n decides whether image should be written to disk (1 = yes, 0 = no)");
    return 0;
  }
  
  /* Find number of CUDA devices (GPUs)
   * and print the name of the first one.
   */
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  printf("Number of CUDA devices: %d\n", n_devices);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  printf("CUDA device name: %s\n" , device_prop.name);
  
  /* Calculate the range in the y - axis such that we preserve the aspect ratio */
  step = (xright - xleft)/XSIZE;
  printf("step = %f\n", step);
  yupper = ycenter + (step * YSIZE)/2;
  ylower = ycenter - (step * YSIZE)/2;
  
  /* Global arrays for iteration counts/pixels
   * One array for the result of the CPU calculation,
   * one for the result of the GPU calculation.
   * (Both are in the host/CPU memory)
   */
  int* pixel_for_cpu = (int*) malloc(sizeof(int) * XSIZE * YSIZE);
  int* pixel_for_gpu = (int*) malloc(sizeof(int) * XSIZE * YSIZE);
  

  /* Perform calculation on CPU */
  double start_cpu = walltime();
  calculate(pixel_for_cpu);
  double end_cpu = walltime();
  
  /* Perform calculations on GPU */
  double start_gpu = walltime();
  calculate_cuda(pixel_for_gpu);
  double end_gpu = walltime();
  
  /* Compare execution times
   * The GPU time also includes the time for memory allocation and transfer
   */
  printf("CPU time: %f s\n" , (end_cpu-start_cpu));
  printf("GPU time: %f s\n" , (end_gpu-start_gpu));
  

  /* Output */
  if (strtol(argv[1], NULL, 10) != 0) {
      output(pixel_for_gpu);
  }
  
  return 0;
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down */
void savebmp(char *name, uchar *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  uchar header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, XSIZE * YSIZE * 3, f);
  fclose(f);
}

/* Given iteration number, set a colour */
void fancycolour(uchar *p, int iter) {
  if (iter == MAXITER);
  else if (iter < 8) { p[0] = 128 + iter * 16; p[1] = p[2] = 0; }
  else if (iter < 24) { p[0] = 255; p[1] = p[2] = (iter - 8) * 16; }
  else if (iter < 160) { p[0] = p[1] = 255 - (iter - 24) * 2; p[2] = 255; }
  else { p[0] = p[1] = (iter - 160) * 2; p[2] = 255 - (iter - 160) * 2; }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(int* pixel){
    unsigned char *buffer = (unsigned char*)calloc(XSIZE * YSIZE * 3, 1);
    for (int i = 0; i < XSIZE; i++) {
      for (int j = 0; j < YSIZE; j++) {
        int p = ((YSIZE - j - 1) * XSIZE + i) * 3;
        fancycolour(buffer + p, pixel[(i + XSIZE * j)]);
      }
    }
    /* write image to disk */
    savebmp("mandel2.bmp", buffer, XSIZE, YSIZE);
    free(buffer);
}
