#define BLOCK_LENGTH 512
#define BLOCK_SHIFT 128
#define BLOCK_FFT 257
#pragma once


#include <iostream>
#include <memory>

#include <vector>
#include <algorithm>
#include <math.h>
#include <complex>
#include <string>
#include <iterator>
#include "mkl_dfti.h"
#include "onnxruntime.hpp"
#include "model2.h"


class DTLN
{
private:
    OnnxENV Env;
    model1 *model_1 = NULL;
    model2 *model_2 = NULL;
    struct model* m2 = NULL; 

    // buffers
    std::vector<float> in_buffer = std::vector<float>(BLOCK_LENGTH, 0);
    std::vector<float> outblock = std::vector<float>(BLOCK_LENGTH, 0);
    std::vector<float> out_buffer = std::vector<float>(BLOCK_LENGTH, 0);


    std::vector<float> in_mag = std::vector<float>(BLOCK_FFT, 0);
    std::vector<float> in_phase= std::vector<float>(BLOCK_FFT, 0);
    std::vector<float> out_mask = std::vector<float>(BLOCK_FFT, 0);
    // 1 * 2 * 128 * 2  
    // model rnn states
    std::vector<float> state_1 = std::vector<float>(BLOCK_LENGTH, 0);
    std::vector<float> state_2 = std::vector<float>(BLOCK_LENGTH, 0);
    //fft
    DFTI_DESCRIPTOR_HANDLE handle = NULL;
    MKL_LONG status;
    std::vector<std::complex<float>> fd = std::vector<std::complex<float>>(BLOCK_FFT, std::complex<float>(0.0f, 0.0f));
    //fftwf_plan rfft = fftwf_plan_dft_r2c_1d(BLOCK_LENGTH, in_buffer.data(), reinterpret_cast<fftwf_complex*>(&fd[0]), FFTW_ESTIMATE);
    //fftwf_plan irfft = fftwf_plan_dft_c2r_1d(BLOCK_LENGTH, reinterpret_cast<fftwf_complex*>(&fd[0]), outblock.data(), FFTW_ESTIMATE);
public:
    DTLN();
    // inference with input vector with size "block_shift" and output vector "block_len"
    bool inference(std::vector<float>& input, std::vector<float>& output);



};