#include <cudnn.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "float32.h"

#define IN_DATA_BYTES (IN_SIZE*sizeof(dtype))
#define OUT_DATA_BYTES (OUT_SIZE*sizeof(dtype))

//function to print out error message from cuDNN calls
#define checkCUDNN(exp) \
  { \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 

int main() {
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnPoolingDescriptor_t pooling_desc;
  //create descriptor handle
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  //initialize descriptor
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         CUDNN_POOLING_MAX,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         3,                       //window height
                                         3,                       //window width
                                         0,                       //vertical padding
                                         0,                       //horizontal padding
                                         1,                       //vertical stride
                                         1));                     //horizontal stride
  
  cudnnTensorDescriptor_t in_desc;
  //create input data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
  //initialize input data descriptor 
  checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,                  //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DTYPE,              //data type (precision)
                                        2,                        //number of images
                                        2,                        //number of channels
                                        10,                       //data height 
                                        10));                     //data width

  cudnnTensorDescriptor_t out_desc;
  //create output data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
  //initialize output data descriptor
  checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,                 //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DTYPE,              //data type (precision)
                                        2,                        //number of images
                                        2,                        //number of channels
                                        8,                        //data height
                                        8));                      //data width

  stype alpha = 1.0f;
  stype beta = 0.0f;
  //GPU data pointers
  dtype *in_data, *out_data;
  //allocate arrays on GPU
  cudaMalloc(&in_data,IN_DATA_BYTES);
  cudaMalloc(&out_data,OUT_DATA_BYTES);
  //copy input data to GPU array
  cudaMemcpy(in_data,input,IN_DATA_BYTES,cudaMemcpyHostToDevice);
  //initize output data on GPU
  cudaMemset(out_data,0,OUT_DATA_BYTES);

  //Call pooling operator
  checkCUDNN(cudnnPoolingForward(cudnn,         //cuDNN context handle
                                 pooling_desc,  //pooling descriptor handle
                                 &alpha,        //alpha scaling factor
                                 in_desc,       //input tensor descriptor
                                 in_data,       //input data pointer to GPU memory
                                 &beta,         //beta scaling factor
                                 out_desc,      //output tensor descriptor
                                 out_data));    //output data pointer from GPU memory

  //allocate array on CPU for output tensor data
  dtype *result = (dtype*)malloc(OUT_DATA_BYTES);
  //copy output data from GPU
  cudaMemcpy(result,out_data,OUT_DATA_BYTES,cudaMemcpyDeviceToHost);

  //loop over and check that the forward pass outputs match expected results (exactly)
  int err = 0;
  for(int i=0; i<OUT_SIZE; i++) {
    if(result[i] != output[i]) {
      std::cout << "Error! Expected " << output[i] << " got " << result[i] << " for idx " << i <<std::endl;
      err++;
    }
  }

  std::cout << "Forward finished with " << err << " errors" << std::endl;

  dtype *in_grad;
  //allocate array on GPU for gradient
  cudaMalloc(&in_grad,IN_DATA_BYTES);
  //initialize output array 
  cudaMemset(in_grad,0,IN_DATA_BYTES);

  //call pooling operator to compute gradient
  checkCUDNN(cudnnPoolingBackward(cudnn,        //cuDNN context handle
                                  pooling_desc, //pooling descriptor handle
                                  &alpha,       //alpha scaling factor
                                  out_desc,     //output tensor descriptor
                                  out_data,     //output tensor pointer to GPU memory
                                  out_desc,     //differential tensor descriptor
                                  out_data,     //differential tensor pointer to GPU memory
                                  in_desc,      //input tensor descriptor
                                  in_data,      //input tensor pointer to GPU memory
                                  &beta,        //beta scaling factor
                                  in_desc,      //gradient tensor descriptor
                                  in_grad));    //gradient tensor pointer to GPU memory

  //allocate array on CPU for gradient tensor data
  dtype *grad = (dtype*)malloc(IN_DATA_BYTES);
  //copy gradient data from GPU
  cudaMemcpy(grad,in_grad,IN_DATA_BYTES,cudaMemcpyDeviceToHost);

  //loop over and check that the forward pass outputs match expected results (within tolerance)
  err = 0;
  for(int i=0; i<IN_SIZE; i++) {
    double diff = std::abs(gradient[i] - grad[i]);
    if(diff > TOL) {
      std::cout << "Error! Expected " << gradient[i] << " got " << grad[i] << " for idx " << i << " diff: " << diff <<std::endl;
      err++;
    }
  }

  std::cout << "Backward finished with " << err << " errors" << std::endl;

  //free CPU arrays
  free(result);
  free(grad);

  //free GPU arrays
  cudaFree(in_data);
  cudaFree(in_grad);
  cudaFree(out_data);

  //free cuDNN descriptors
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  cudnnDestroy(cudnn);
  
  return 0;
}
