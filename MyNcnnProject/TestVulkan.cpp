#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "net.h"


#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
//#define MODEL_CHANNEL ncnn::Mat::PIXEL_GRAY
#define MODEL_CHANNEL ncnn::Mat::PIXEL_BGR
void TestVulkan(void)
{
	printf("TestVulkan\n");

	/*** initialize when app starts ***/
	ncnn::create_gpu_instance();
	printf("GPU Count: %d\n", ncnn::get_gpu_count());

	/*** Load ncnn model (probably, need this only once) ***/
	ncnn::Net net;
	net.use_vulkan_compute = 1;		// it looks like, this line must be the first
	net.load_param(RESOURCE_DIR"ncnn_mobilenet.param");
	net.load_model(RESOURCE_DIR"ncnn_mobilenet.bin");
	
	/*** Read image using OpenCV ***/
	cv::Mat image = cv::imread(RESOURCE_DIR"parrot.jpg", (MODEL_CHANNEL == ncnn::Mat::PIXEL_GRAY) ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
	cv::imshow("Display", image);

	/*** Prepare input image Mat for ncnn ***/
	ncnn::Mat ncnnMat = ncnn::Mat::from_pixels_resize(image.data, MODEL_CHANNEL, image.cols, image.rows, MODEL_WIDTH, MODEL_HEIGHT);
	float mean[3] = { 128.f, 128.f, 128.f };
	float norm[3] = { 1 / 128.f, 1 / 128.f, 1 / 128.f };
	ncnnMat.substract_mean_normalize(mean, norm);

	/*** Prepare inference ***/
	ncnn::Extractor ex = net.create_extractor();
	ex.set_vulkan_compute(true);
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input("data", ncnnMat);

	/*** Run inference ***/
	ncnn::Mat ncnnOut;
	auto t0 = std::chrono::system_clock::now();
	ex.extract("mobilenetv20_output_flatten0_reshape0", ncnnOut);
	auto t1 = std::chrono::system_clock::now();
	
	/*** Retrieve result ***/
	int outputNum = ncnnOut.w;
	//float *results = new float[outputNum];
	float results[1000];
	for (int i = 0; i < outputNum; i++) {
		results[i] = ((float*)ncnnOut.data)[i];
	}

	float maxScore = 0;
	int maxIndex = -1;
	for (int i = 0; i < outputNum; i++) {
		if (maxScore < results[i]) {
			maxScore = results[i];
			maxIndex = i;
		}
	}

	printf("Result = %d (%.3f)\n", maxIndex, maxScore);
	double inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
	printf("Inference time: %.2lf [msec]\n", inferenceTime);
	//delete[] results;

	/*** deinitialize when app exits ***/
	ncnn::destroy_gpu_instance();

	// cv::waitKey(0);
}
