#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx/onnx.hpp"
namespace yolov10 {
void set_input_image_internal(const cv::Mat& image, float* data,
                              const std::vector<float>& mean,
                              const std::vector<float>& scale, bool btrans) {
  int size = image.rows * image.cols;
  float* channel0_ptr = data;
  float* channel1_ptr = data + size;
  float* channel2_ptr = data + size * 2;
  // BGR->RGB (maybe) and HWC->CHW
  for (int h = 0; h < image.rows; h++) {
    for (int w = 0; w < image.cols; w++) {
      channel0_ptr[h * image.cols + w] =
          (image.at<cv::Vec3b>(h, w)[0] - mean[0]) * scale[0];
      channel1_ptr[h * image.cols + w] =
          (image.at<cv::Vec3b>(h, w)[1] - mean[1]) * scale[1];
      channel2_ptr[h * image.cols + w] =
          (image.at<cv::Vec3b>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}
void set_input_image(const cv::Mat& image, float* data,
                     const std::vector<float>& mean,
                     const std::vector<float>& scale) {
  return set_input_image_internal(image, data, mean, scale, false);
}
struct Result {
  struct BoundingBox {
    int label;
    float score;
    std::vector<float> box;
  };
  std::vector<BoundingBox> bboxes;
};
namespace yolov10_helper {
inline std::vector<int> b = {
    144, 89,  30,  3,   16,  69,  237, 54,  4,   89,  15,  141, 87,  65,
    118, 150, 117, 119, 19,  90,  33,  53,  39,  11,  228, 93,  40,  164,
    46,  228, 48,  163, 114, 182, 232, 103, 21,  49,  116, 54,  62,  160,
    159, 163, 212, 117, 237, 169, 94,  16,  79,  124, 68,  154, 190, 70,
    203, 178, 64,  55,  206, 79,  25,  230, 43,  52,  255, 230, 116, 3,
    135, 175, 78,  158, 254, 50,  161, 223, 204, 108, 63};
inline std::vector<int> g = {
    246, 80,  103, 0,   134, 12,  197, 233, 7,   31,  118, 88,  161, 221,
    236, 228, 71,  81,  26,  143, 188, 5,   154, 6,   152, 224, 39,  126,
    196, 216, 177, 149, 161, 65,  192, 133, 5,   254, 151, 66,  158, 117,
    193, 173, 56,  252, 5,   197, 37,  143, 131, 220, 229, 73,  176, 60,
    124, 46,  36,  44,  200, 92,  126, 216, 248, 151, 189, 162, 135, 145,
    244, 158, 135, 188, 34,  33,  99,  10,  146, 107, 139};
inline std::vector<int> r = {
    100, 9,   243, 216, 240, 65,  11,  32,  124, 164, 27,  14,  83,  143,
    49,  27,  59,  131, 138, 184, 178, 176, 73,  16,  10,  226, 189, 30,
    150, 31,  126, 95,  144, 13,  205, 128, 226, 39,  158, 12,  202, 155,
    210, 255, 250, 106, 93,  25,  56,  36,  51,  20,  134, 108, 120, 41,
    118, 163, 162, 55,  122, 160, 89,  173, 240, 218, 187, 150, 231, 78,
    177, 184, 160, 246, 36,  11,  152, 221, 108, 249, 216};
inline std::vector<std::string> classes{
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};
cv::Scalar getColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
}
}  // namespace yolov10_helper
Image show_reusult(Image& image, const Result& result) {
  for (auto& res : result.bboxes) {
    int label = res.label;
    auto& box = res.box;
    cv::rectangle(image, cv::Point(box[0], box[1]),
                  cv::Point(box[2], box[3]),
                  cv::Scalar(yolov10_helper::b[label], yolov10_helper::g[label],
                             yolov10_helper::r[label]),
                  2, 1, 0);
    cv::putText(
        image, yolov10_helper::classes[label] + " " + std::to_string(res.score),
        cv::Point(box[0] + 5, box[1] + 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.4,
        cv::Scalar(yolov10_helper::b[label], yolov10_helper::g[label],
                   yolov10_helper::r[label]),
        1, 4);
  }
  cv::putText(image, std::string("YOLOV10"), cv::Point(20, image.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 1);
  return image;
}
}  // namespace yolov10
class Yolov10 : public Model {
 public:
  Yolov10() {}
  virtual ~Yolov10() {}
  void init(const Config& config) {
    Model::init(config);
    CONFIG_GET(config, float, thresh, "confidence_threshold")
    conf_thresh = thresh;
    output_shapes_.resize(output_tensor_size);
    output_shapes_[0] = session_->get_output_shape_from_model(0);
  }
  void preprocess(const std::vector<Image>& images) override {
    std::vector<float>& input_data_0 = session_->get_input(0);
    std::vector<int64_t>& input_shape_0 = session_->get_input_shape(0);
    int batch_size = images.size();
    input_shape_0[0] = batch_size;
    int64_t total_number_elements =
        std::accumulate(input_shape_0.begin(), input_shape_0.end(), int64_t{1},
                        std::multiplies<int64_t>());
    input_data_0.resize(size_t(total_number_elements));
    auto channel = input_shape_0[1];
    auto width = input_shape_0[2];
    auto height = input_shape_0[3];
    auto batch_element_size = channel * height * width;
    auto size = cv::Size((int)width, (int)height);
    for (auto index = 0; index < batch_size; ++index) {
      cv::Mat resized_image;
      cv::resize(images[index], resized_image,
                 cv::Size(size.height, size.width));
      yolov10::set_input_image(
          resized_image, input_data_0.data() + batch_size * index,
          std::vector<float>{0, 0, 0},
          std::vector<float>{0.00392156862745098f, 0.00392156862745098f,
                             0.00392156862745098f});
    }
  }
  std::vector<Image> postprocess(const std::vector<Image>& images) override {
    auto batch_size = images.size();
    std::vector<yolov10::Result> results;
    for (auto index = 0; index < batch_size; ++index) {
      results.emplace_back(postprocess_one(index));
    }
    std::vector<Image> image_results;
    for (auto index = 0; index < batch_size; ++index) {
      auto result = results[index];
      auto image = images[index];
      image_results.push_back(yolov10::show_reusult(image, result));
    }
    return image_results;
  }
  yolov10::Result postprocess_one(int idx) {
    float* output_ptr = session_->get_output(0);
    int nums = 300;
    int length = 6;
    std::vector<yolov10::Result::BoundingBox> results;
    for (int i = 0; i < nums; i = i + length) {
      float* resPtr = output_ptr + i;
      if (resPtr[4] > conf_thresh) {
        yolov10::Result::BoundingBox result;
        result.box.resize(4);
        result.box[0] = resPtr[0];
        result.box[1] = resPtr[1];
        result.box[2] = resPtr[2];
        result.box[3] = resPtr[3];
        result.score = resPtr[4];
        result.label = int(resPtr[5]);
        results.push_back(result);
      }
    }
    return yolov10::Result{results};
  }

 private:
  int output_tensor_size{1};
  float conf_thresh{0.f};
  int num_classes{80};
  std::vector<std::vector<int64_t>> output_shapes_;
};
REGISTER_MODEL(yolov10, Yolov10)