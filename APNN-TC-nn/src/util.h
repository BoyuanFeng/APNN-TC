#ifndef util_h
#define util_h


void init_all_one_matrix(float* data, int size){
    #pragma omp parallel for
    for (int i = 0; i < size; i++){
        data[i] = 1.0f;
    }
}

#endif