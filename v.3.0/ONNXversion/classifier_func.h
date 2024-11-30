#ifndef __CLASSIFIER_FUNC_H__
#define __CLASSIFIER_FUNC_H__


#ifdef __cplusplus
 #define UNICODE
 #include <onnxruntime_cxx_api.h>
 #include <array>
 #include <cmath>
 #include <iostream>
 #include <algorithm>

/// ========= Model parameters ============
 #define MODEL_NAME "classifier0.onnx"
 #define INPUT_LAYER "classifier_vec_input"
 #define OUTPUT_LAYER "norm_layer"
 static constexpr const int width_in_ = 9;
 static constexpr const int width_out_ = 35;
/// =======================================
#endif


#ifdef __cplusplus
 extern "C" {
#endif

void ClassifierRun(float* input_f_,float* results_f_);

int argmax(float* arr,float* prob, int len);

#ifdef __cplusplus
 }
#endif


#endif
