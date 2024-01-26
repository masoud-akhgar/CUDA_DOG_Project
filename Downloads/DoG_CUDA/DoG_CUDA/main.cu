#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <iostream>
#include <cmath>
#include<opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
//#include <opencv2/core.hpp>

#define TILE_WIDTH 32
#define BLOCK_SIZE 16

using namespace std;
using namespace cv;


__constant__ float filter1a[9];
__constant__ float filter2a[49];


Mat getGaussian(int height, int width, double sigma) {
    nvtxMark("Get Gaussian");
    nvtxRangePush(__FUNCTION__);

    Mat kernel = Mat(height, width, CV_32FC1);
//    Mat kernel = Mat(height, width, CV_8UC1);
    float sum = 0.0;
    float r, s = 2.0 * sigma * sigma;
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            r = sqrt(i * i + j * j);
            kernel.at<float>(i, j) = exp(-(r * r) / (s)) / (M_PI * s);
            sum += kernel.at<float>(i, j);
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            kernel.at<float>(i, j) /= sum;
        }
    }

    nvtxRangePop();
    return kernel;
}

void getGaussianConst(float kernel[], int height, int width, double sigma) {
    nvtxMark("Get Gaussian Const");
    nvtxRangePush(__FUNCTION__);

    float sum = 0.0;
    float r, s = 2.0 * sigma * sigma;
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            r = sqrt(i * i + j * j);
            kernel[j * width + i] = exp(-(r * r) / (s)) / (M_PI * s);
            sum += kernel[j * width + i];
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            kernel[j * width + i] /= sum;
        }
    }

    nvtxRangePop();
}

__global__ void convFilterGPU(const float *const sourceGPU, const int source_step,
                              const float *const filterGPU, const int filter_step,
                              const float *const out, const int out_step,
                              const int window, const int w, const int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w - window - 1 && Row < h - window - 1) {
        float sum = 0;
        for (int k = 0; k < window; ++k) {
            for (int l = 0; l < window; ++l) {
                sum += ((float *) ((unsigned char *) sourceGPU + (Row + l) * source_step))[Col + k] *
                       ((float *) ((unsigned char *) filterGPU + l * filter_step))[k];
//                sum += filterGPU.at<float>(k, l) * sourceGPU.at<float>(Col+k-1, Row+l-1);
            }
        }
        ((float *) ((unsigned char *) out + (Row + int(window / 2)) * out_step))[Col + int(window / 2)] = sum;
//        sourceGPU.at<float>(Col+int(window/2), Row+int(window/2)) = sum;
    }
}

__global__ void convFilterTiledGPU(const float *const sourceGPU, const int source_step,
                                   const float *const filterGPU, const int filter_step,
                                   const float *const out, const int out_step,
                                   const int window, const int w, const int h) {

    __shared__ float box[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;
    int boxCol = bx * blockDim.x + 2 * tx;
    int boxRow = by * blockDim.y + 2 * ty;

    if (Col < w - window - 1 && Row < h - window - 1) {
        box[2 * ty][2 * tx] = ((float *) ((unsigned char *) sourceGPU + boxRow * source_step))[boxCol];
        box[2 * ty + 1][2 * tx] = ((float *) ((unsigned char *) sourceGPU + (boxRow + 1) * source_step))[boxCol];
        box[2 * ty][2 * tx + 1] = ((float *) ((unsigned char *) sourceGPU + boxRow * source_step))[boxCol + 1];
        box[2 * ty + 1][2 * tx + 1] = ((float *) ((unsigned char *) sourceGPU + (boxRow + 1) * source_step))[boxCol + 1];
        __syncthreads();

        float sum = 0;
        for (int i = 0; i < window; ++i)
            for (int j = 0; j < window; ++j)
                sum += box[ty + j][tx + i] *
                       ((float *) ((unsigned char *) filterGPU + j * filter_step))[i];

        ((float *) ((unsigned char *) out + (Row + int(window / 2)) * out_step))[Col + int(window / 2)] = sum;
    }
}

// window2 > window1
__global__ void convFilterTiledGPUWithConstant(const float *const sourceGPU, const int source_step,
//                                               const float *const dogGPU, const int dog_step,
                                               const float *const out1, const int out1_step,
                                               const float *const out2, const int out2_step,
                                               const int window1, const int window2, const int w, const int h) {

    __shared__ float box[TILE_WIDTH][TILE_WIDTH];
//    __shared__ float result1[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ float result2[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;
    int boxCol = bx * blockDim.x + 2 * tx;
    int boxRow = by * blockDim.y + 2 * ty;

//    result1[ty][tx] = result2[ty][tx] = ((float *) ((unsigned char *) dogGPU + Row * dog_step))[Col];

    if (Col < w - window2 - 1 && Row < h - window2 - 1) {
        box[2 * ty][2 * tx] = ((float *) ((unsigned char *) sourceGPU + boxRow * source_step))[boxCol];
        box[2 * ty + 1][2 * tx] = ((float *) ((unsigned char *) sourceGPU + (boxRow + 1) * source_step))[boxCol];
        box[2 * ty][2 * tx + 1] = ((float *) ((unsigned char *) sourceGPU + boxRow * source_step))[boxCol + 1];
        box[2 * ty + 1][2 * tx + 1] = ((float *) ((unsigned char *) sourceGPU + (boxRow + 1) * source_step))[boxCol + 1];
        __syncthreads();

        float sum1 = 0;
        float sum2 = 0;
        for (int i = 0; i < window2; ++i)
            for (int j = 0; j < window2; ++j) {
                if (i < window1 && j < window1)
                    sum1 += box[ty + j][tx + i] * filter1a[j * window1 + i];
                sum2 += box[ty + j][tx + i] * filter2a[j * window2 + i];
            }

        ((float *) ((unsigned char *) out1 + (Row + int(window1 / 2)) * out1_step))[Col + int(window1 / 2)] = sum1;
        ((float *) ((unsigned char *) out2 + (Row + int(window2 / 2)) * out2_step))[Col + int(window2 / 2)] = sum2;
//    ((float *) ((unsigned char *) dogGPU + Row * dog_step))[Col] =
//            result1[ty][tx] - result2[ty][tx];
    }
}

__global__ void dogGPU(const float *const source1GPU, const int source1_step,
                       const float *const source2GPU, const int source2_step,
                       const int w, const int h) {

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        ((float *) ((unsigned char *) source1GPU + Row * source1_step))[Col] -= ((float *) (
                (unsigned char *) source2GPU + Row * source2_step))[Col];
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void run_app() {
    /*int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Warps Size: %d\n",
               prop.warpSize);
        printf("  Max Grid Size: %d,\t%d,\t%d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Threads Dim: %d,\t%d,\t%d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  L2 Cache Size: %d\n",
               prop.l2CacheSize);
        printf("  Max Threads per Block: %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  Max Blocks per SM: %d\n",
               prop.maxBlocksPerMultiProcessor);
        printf("  Total amount of constant memory: %zu\n",
               prop.totalConstMem);
        printf("  Total amount of global memory: %zu\n",
               prop.totalGlobalMem);
        printf("  Total registers per block: %d\n",
               prop.regsPerBlock);
        printf("  Total amount of shared memory per block: %zu\n",
               prop.sharedMemPerBlock);;
        printf("  Total registers per SM: %d\n",
               prop.regsPerMultiprocessor);
        printf("  Total amount of shared memory per SM: %zu\n",
               prop.sharedMemPerMultiprocessor);
        printf("  Number of multiprocessors: %d\n",
               prop.multiProcessorCount);
        printf("  Major revision number: %d\n",
               prop.major);
        printf("  Number of cores: %d\n",
               getSPcores(prop));
    }*/

    Mat img = imread("/home/alimgh/Documents/University/Multicore/CUDA/DoG_CUDA/bigImg.jpg", IMREAD_GRAYSCALE);

    if (img.empty()) {
        printf("No image data \n");
        exit(-1);
    }

    Mat img_f, source1, source2;
    img.convertTo(img_f, CV_32FC1);

//    Mat filter1 = getGaussian(3, 3, 16);
//    Mat filter2 = getGaussian(7, 7, 32);

    float hfilter1a[9];
    float hfilter2a[49];
    getGaussianConst(hfilter1a, 3, 3, 16);
    getGaussianConst(hfilter2a, 7, 7, 32);
    cudaMemcpyToSymbol(filter1a, &hfilter1a, 9 * sizeof(float));
    cudaMemcpyToSymbol(filter2a, &hfilter2a, 49 * sizeof(float));

//    cv::cuda::GpuMat filter1GPU{filter1.rows, filter1.cols, CV_32FC1};
//    cv::cuda::GpuMat filter2GPU{filter2.rows, filter2.cols, CV_32FC1};
    cv::cuda::GpuMat source1GPU{img.rows, img.cols, CV_32FC1};
//    cv::cuda::GpuMat source2GPU{img.rows, img.cols, CV_32FC1};
    cv::cuda::GpuMat out1{img.rows, img.cols, CV_32FC1};
    cv::cuda::GpuMat out2{img.rows, img.cols, CV_32FC1};
//    cv::cuda::GpuMat out3{img.rows, img.cols, CV_32FC1};

//    filter1GPU.upload(filter1);
//    filter2GPU.upload(filter2);

    source1 = img_f.clone();
//    source2 = img_f.clone();
    source1GPU.upload(source1);
//    source2GPU.upload(source2);

//    Mat r1 = convFilterSerial(img_f, filter1, 3);
//    Mat r2 = convFilterSerial(img_f, filter2, 7);

    dim3 DimGrid((source1.cols - 1) / 16 + 1, (source1.rows - 1) / 16 + 1, 1);
    dim3 DimBlock(16, 16, 1);
//    convFilterGPU<<<DimGrid, DimBlock>>>((float *) source1GPU.data, source1GPU.step,
//                                         (float *) filter1GPU.data, filter1GPU.step,
//                                         (float *) out1.data, out1.step,
//                                         3, source1.cols, source1.rows);

//    convFilterTiledGPU<<<DimGrid, DimBlock>>>((float *) source1GPU.data, source1GPU.step,
//                                              (float *) filter1GPU.data, filter1GPU.step,
//                                              (float *) out1.data, out1.step,
//                                              3, source1.cols, source1.rows);

    convFilterTiledGPUWithConstant<<<DimGrid, DimBlock>>>((float *) source1GPU.data, source1GPU.step,
//                                                          (float *) out3.data, out3.step,
                                                          (float *) out1.data, out1.step,
                                                          (float *) out2.data, out2.step,
                                                          3, 7, source1.cols, source1.rows);


//    Mat r1(img.rows, img.cols, CV_32FC1);
//    out1.download(r1);
//    imwrite("/home/alimgh/Documents/University/Multicore/CUDA/DoG_CUDA/g4_1.jpg", r1);

//    convFilterGPU<<<DimGrid, DimBlock>>>((float *) source2GPU.data, source2GPU.step,
//                                         (float *) filter2GPU.data, filter2GPU.step,
//                                         (float *) out2.data, out2.step,
//                                         7, source2.cols, source2.rows);

//    convFilterTiledGPU<<<DimGrid, DimBlock>>>((float *) source2GPU.data, source2GPU.step,
//                                         (float *) filter2GPU.data, filter2GPU.step,
//                                         (float *) out2.data, out2.step,
//                                         7, source2.cols, source2.rows);

//    out2.download(r1);
//    imwrite("/home/alimgh/Documents/University/Multicore/CUDA/DoG_CUDA/g4_2.jpg", r1);

    dogGPU<<<DimGrid, DimBlock>>>((float *) out1.data, out1.step,
                                  (float *) out2.data, out2.step,
                                  out1.cols, out1.rows);


    Mat r1(img.rows, img.cols, CV_32FC1);
    out1.download(r1);
    imwrite("/home/alimgh/Documents/University/Multicore/CUDA/DoG_CUDA/big_dog.jpg", r1);

//    dog(r1, r2);

//    namedWindow("original_image",WINDOW_AUTOSIZE);
//    imshow("original_image",img);
//    namedWindow("gray_image",WINDOW_AUTOSIZE);
//    imshow("gray_image",grayImg);
//    waitKey(0);
}

int main() {

    nvtxMark("Main Thread");
    nvtxRangePush(__FUNCTION__);
    run_app();
    nvtxRangePop();

    return 0;
}
