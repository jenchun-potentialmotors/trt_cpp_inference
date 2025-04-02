#include "include/Image.h"
#include "include/Tensor.h"
#include "include/TensorRTEngine.h" // 添加 TensorRTEngine 的標頭檔
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ModelData {
const std::string ENGINE_PATH = "best.trt";
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <annotation_file>"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string ann_path = argv[2];

  TensorrtEngine trtInference(engine_path);
  if (!trtInference.initialize()) {
    return -1;
  }

  std::ifstream ann_file(ann_path);
  if (!ann_file.is_open()) {
    std::cerr << "Error: Unable to open annotation file: " << ann_path
              << std::endl;
    return -1;
  }

  int num_correct = 0;
  int num_total = 0;
  double totalInferenceTime = 0.0;

  std::cout << "\n--- Inference Results ---" << std::endl;
  std::cout << std::left << std::setw(60) << "Image Name" << std::right
            << std::setw(10) << "Label" << std::right << std::setw(10)
            << "Prediction" << std::right << std::setw(15) << "Confidence"
            << std::right << std::setw(20) << "Inference Time (ms)"
            << std::endl;
  std::cout << std::string(115, '-') << std::endl;

  std::string line;
  while (std::getline(ann_file, line)) {
    num_total++;

    std::istringstream ss(line);
    std::string image_path, label;
    ss >> image_path >> label;

    auto oros_image = oros::Image(image_path);
    if (oros_image.empty()) {
      std::cerr << "Warning: Unable to load image: " << image_path << std::endl;
      continue;
    }

    oros_image = oros_image.resize(298, 168);
    auto tensor = oros::Tensor(oros_image, {123.675f, 116.28f, 103.53f},
                               {58.395f, 57.12f, 57.375f});
    auto buffer = static_cast<std::vector<float>>(tensor);

    std::vector<float> outputs;

    auto start = std::chrono::high_resolution_clock::now();
    if (!trtInference.executeInference(buffer, outputs)) {
      std::cerr << "Error: Inference failed for image: " << image_path
                << std::endl;
      continue;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double inferenceTime =
        std::chrono::duration<double, std::milli>(end - start).count();
    totalInferenceTime += inferenceTime;

    auto maxElementPtr = std::max_element(outputs.begin(), outputs.end());
    int maxIndex = std::distance(outputs.begin(), maxElementPtr);
    float maxValue = *maxElementPtr;

    std::filesystem::path image_file_path(image_path);
    std::string image_name = image_file_path.filename().string();

    std::cout << std::left << std::setw(60) << image_name << std::right
              << std::setw(10) << label << std::right << std::setw(10)
              << maxIndex << std::right << std::setw(15) << std::fixed
              << std::setprecision(4) << maxValue << std::right << std::setw(20)
              << std::fixed << std::setprecision(2) << inferenceTime
              << std::endl;

    if (maxIndex == stoi(label)) {
      num_correct++;
    }
  }

  std::cout << std::endl;
  std::cout << "--- Summary ---" << std::endl;
  std::cout << "Total Images: " << num_total << std::endl;
  std::cout << "Correct Predictions: " << num_correct << std::endl;
  std::cout << "Accuracy: " << std::fixed << std::setprecision(4)
            << float(num_correct) / float(num_total) << std::endl;
  std::cout << "Avg Inference Time: " << std::fixed << std::setprecision(4)
            << totalInferenceTime / num_total << " ms" << std::endl;

  return 0;
}