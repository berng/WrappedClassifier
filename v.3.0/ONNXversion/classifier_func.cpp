#include "classifier_func.h"

extern "C"
 void ClassifierRun(float* input_f_, float* results_f_) 
 {

   std::array<float, width_in_> input_{};
   std::array<float, width_out_> results_{};
   int64_t result_{0};
   Ort::Env env;
   Ort::Session session_{env, MODEL_NAME, Ort::SessionOptions{nullptr}};

   Ort::Value input_tensor_{nullptr};
   std::array<int64_t, 2> input_shape_{1, width_in_};

   Ort::Value output_tensor_{nullptr};
   std::array<int64_t, 2> output_shape_{1, width_out_};


    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_.data(), input_.size(),
                                                    input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                     output_shape_.data(), output_shape_.size());

    const char* input_names[] = {INPUT_LAYER};
    const char* output_names[] = {OUTPUT_LAYER};

    for(int i=0;i<width_in_;i++)
     input_[i]=input_f_[i];

    Ort::RunOptions run_options;
    session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    for(int i=0;i<width_out_;i++)
     results_f_[i]=(float)results_[i];
//    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return;
 }


extern "C"
 int argmax(float* arr, float* prob, int width_out_)
 {
  int pos=-1;
  float max=-1;
  for(int i=0;i<width_out_;i++)
   if (arr[i]>max)
    { max=arr[i]; pos=i; }
  if (prob)
   *prob=max;
  return pos;
 }
