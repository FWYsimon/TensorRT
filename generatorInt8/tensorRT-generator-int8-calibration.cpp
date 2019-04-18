



#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <unordered_map>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"

#include <fstream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <memory>

#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <varargs.h>
#include <pa_file\pa_file.h>
#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvinfer_plugin.lib")
#pragma comment(lib, "nvonnxparser.lib")
#pragma comment(lib, "nvparsers.lib")
#pragma comment(lib, "cudart.lib")

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace std;
using namespace cv;

void errorExit(const char* fmt, ...){
	va_list vl;
	va_start(vl, fmt);

	printf("error exit.\n");
	vprintf(fmt, vl);

	system("pause");
	exit(0);
}


#define CHECK(status)                             \
    do                                            \
	    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
		        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
		        }                                         \
	    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
		: reportableSeverity(severity)
	{
	}

	void log(Severity severity, const char* msg) override
	{
		// suppress messages with severity enum value greater than the reportable
		if (severity > reportableSeverity)
			return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
		case Severity::kERROR: std::cerr << "ERROR: "; break;
		case Severity::kWARNING: std::cerr << "WARNING: "; break;
		case Severity::kINFO: std::cerr << "INFO: "; break;
		default: std::cerr << "UNKNOWN: "; break;
		}
		std::cerr << msg << std::endl;
	}

	Severity reportableSeverity;
}gLogger;

enum MODE
{
	kFP32,
	kFP16,
	kINT8,
	kUNKNOWN
};

struct Tensor{
	float* ptr = nullptr;
	int dims[4];

	Tensor(){
		setShape(0, 0, 0, 0);
	}

	Tensor(const Tensor& other){
		from(other.ptr, other.dims[0], other.dims[1], other.dims[2], other.dims[3]);
	}

	Tensor& operator=(const Tensor& other){
		from(other.ptr, other.dims[0], other.dims[1], other.dims[2], other.dims[3]);
		return *this;
	}

	~Tensor(){
		release();
	}

	void from(const float* ptr, int axis0, int axis1, int axis2, int axis3){
		if (this->count() != axis0 * axis1 * axis2 * axis3)
			create(axis0, axis1, axis2, axis3);
		else
			setShape(axis0, axis1, axis2, axis3);
		memcpy(this->ptr, ptr, sizeof(float) * this->count());
	}

	void setShape(int axis0, int axis1, int axis2, int axis3){
		this->dims[0] = axis0;
		this->dims[1] = axis1;
		this->dims[2] = axis2;
		this->dims[3] = axis3;
	}

	void release(){
		if (ptr){
			delete[] ptr;
			ptr = nullptr;
		}
		setShape(0, 0, 0, 0);
	}

	void transposeInplace(int axis0, int axis1, int axis2, int axis3){
		Tensor t = transpose(axis0, axis1, axis2, axis3);
		setShape(t.dims[0], t.dims[1], t.dims[2], t.dims[3]);
		memcpy(ptr, t.ptr, count() * sizeof(float));
	}

	Tensor transpose(int axis0, int axis1, int axis2, int axis3){

		int muls[] = { count(1), count(2), count(3), 1 };
		Tensor t;
		t.create(dims[axis0], dims[axis1], dims[axis2], dims[axis3]);

		for (int a0 = 0; a0 < dims[axis0]; ++a0){
			for (int a1 = 0; a1 < dims[axis1]; ++a1){
				for (int a2 = 0; a2 < dims[axis2]; ++a2)
					for (int a3 = 0; a3 < dims[axis3]; ++a3)
						t.ptr[a0 * t.count(1) + a1 * t.count(2) + a2 * t.count(3) + a3] = ptr[a0 * muls[axis0] + a1 * muls[axis1] + a2 * muls[axis2] + a3 * muls[axis3]];
			}
		}
		return t;
	}

	int count(int axis = 0) const{
		int v = 1;
		for (int i = axis; i < 4; ++i)
			v *= dims[i];
		return v;
	}

	void create(int axis0, int axis1, int axis2, int axis3){
		release();
		setShape(axis0, axis1, axis2, axis3);

		ptr = new float[count()];
		memset(ptr, 0, sizeof(float) * count());
	}
};

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:

	Int8EntropyCalibrator(Size size, const string& imageFromDirectory){

		this->size_ = size;
		this->imageFromDirectory_ = imageFromDirectory;
		paFindFiles(this->imageFromDirectory_.c_str(), this->allimgs_, "*.jpg;*.png;*.bmp;*.ppm;*.tif;*.jpeg", true);

		int batchSize = 1;
		batchCudaSize_ = batchSize * 3 * size.height * size.width;
		CHECK(cudaMalloc(&batchCudaMemory_, batchCudaSize_ * sizeof(float)));
	}

	virtual ~Int8EntropyCalibrator(){
		CHECK(cudaFree(batchCudaMemory_));
	}

	int getBatchSize() const override{
		return 1;
	}

	bool next(){
		if (cursor_ >= allimgs_.size()){
			printf("\n");
			return false;
		}

		Mat im = imread(allimgs_[cursor_++]);
		if (im.empty()){
			printf("\n");
			return false;
		}

		for (int i = 0; i < lastprintLength_; ++i)
			printf("\b");

		char printline[100];
		sprintf(printline, "process image %d / %d", cursor_, allimgs_.size());
		lastprintLength_ = strlen(printline);

		printf("%s", printline);
		resize(im, im, size_);
		//cvtColor(im, im, CV_BGR2RGB);
		im.convertTo(im, CV_32F, 1 / 128.0, -127.5 / 128.0);

		//copy from image
		caffeTensor.from(im.ptr<float>(0), 1, im.rows, im.cols, im.channels());

		//transpose to caffe format
		caffeTensor.transposeInplace(0, 3, 1, 2);
		CHECK(caffeTensor.count() != batchCudaSize_);
		CHECK(cudaMemcpy(this->batchCudaMemory_, caffeTensor.ptr, caffeTensor.count() * sizeof(float), cudaMemcpyHostToDevice));
		return true;
	}

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override{
		if (!next()) return false;

		bindings[0] = batchCudaMemory_;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override {
		length = 0;
		return nullptr;

		//return paReadFile("CalibrationTableresnet50", &length);
	}

	virtual void writeCalibrationCache(const void* cache, size_t length) override{
	}

private:
	int lastprintLength_ = 0;
	PaVfiles allimgs_;
	size_t batchCudaSize_ = 0;
	void* batchCudaMemory_ = nullptr;
	string imageFromDirectory_;
	int cursor_ = 0;
	Size size_;
	Tensor caffeTensor;
};

void caffeToTRTModel(
	int maxBatchSize,
	const std::string& deployFile,           // Name for caffe prototxt
	const std::string& modelFile,            // Name for model
	const std::vector<std::string>& outputs, // Network outputs
	const string& imageDirectory,			 // train int8 images
	Size networkInputSize,
	const string& int8ModelSaveFile)		 // Output stream for the TensorRT model
{
	// Create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// Parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	//parser->setPluginFactory(pluginFactory);
	nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;

	//std::cout << "Begin parsing model..." << std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(
		deployFile.c_str(),
		modelFile.c_str(),
		*network,
		dataType);
	//std::cout << "End parsing model..." << std::endl;

	// Specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	size_t _workSize = 1 << 30;
	builder->setMaxWorkspaceSize(_workSize);
	builder->setMaxBatchSize(maxBatchSize);

	// Calibrator life time needs to last until after the engine is built.
	std::unique_ptr<IInt8Calibrator> calibrator;
	ICudaEngine* engine;

	//std::cout << "Using Entropy Calibrator" << std::endl;
	calibrator.reset(new Int8EntropyCalibrator(networkInputSize, imageDirectory));
	builder->setInt8Mode(true);
	builder->setInt8Calibrator(calibrator.get());
	//std::cout << "Begin building engine..." << std::endl;
	engine = builder->buildCudaEngine(*network);
	assert(engine);
	//std::cout << "End building engine..." << std::endl;

	// Once the engine is built. Its safe to destroy the calibrator.
	calibrator.reset();

	// We don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	IHostMemory* seril = engine->serialize();
	paWriteToFile(int8ModelSaveFile.c_str(), seril->data(), seril->size());

	seril->destroy();
	engine->destroy();
	builder->destroy();
}

void generatorRefineDet(){
	std::vector<std::string> outputblobs
	{ "odm_loc", "odm_conf_flatten", "arm_priorbox", "arm_conf_flatten", "arm_loc" };

#if 0
	caffeToTRTModel(
		2,
		"检测器-refdet320/deploy.prototxt",
		"检测器-refdet320/refinedet_vgg16_refinedet_vgg16_320x320_iter_100000.caffemodel",
		outputblobs,
		"batch_images", Size(320, 320), "ref320_b2_1080ti.trtmodel");
#else
	caffeToTRTModel(
		4,
		"检测器-refdet/deploy.prototxt",
		"检测器-refdet/refinedet_vgg16_refinedet_vgg16_512x512_iter_40000.caffemodel",
		outputblobs,
		"batch_images", Size(512, 512), "ref512_b4_1080ti.trtmodel");
#endif
}

void generatorSZClassifier(){
	std::vector<std::string> outputblobs
	{ "EmbedNetworkResnetnLinearnfc5n506" };

	caffeToTRTModel(
		24,
		"分类器-sz/caffe_target_iplugin.prototxt",
		"分类器-sz/caffe_target.caffemodel",
		outputblobs,
		"batch_images", Size(94, 94), "9494_b24_classifier_1080ti.trtmodel");
}

void generatorTPClassifier(){
	std::vector<std::string> outputblobs
	{ "prob" };

	caffeToTRTModel(
		24,
		"I:/code-research/suanfa-news-bigmodel-trace-base.git/bin/libs/left/deploy.prototxt",
		"I:/code-research/suanfa-news-bigmodel-trace-base.git/bin/libs/left/20190329_resnet50softmax_withdataAugment_YuAoleftAndBg__iter_4500.caffemodel",
		outputblobs,
		"batch_images", Size(112, 112), "res50left.trtmodel");
}

void generatorTPRClassifier(){
	std::vector<std::string> outputblobs
	{ "fc5" };


	cudaSetDevice(0);
	caffeToTRTModel(
		24,
		"I:/code-research/suanfa-news-bigmodel-trace-base.git/test-projects/shenzhen_Integrate/tensorRT-generatormodel/分类器1350/deploy.prototxt",
		"I:/code-research/suanfa-news-bigmodel-trace-base.git/test-projects/shenzhen_Integrate/tensorRT-generatormodel/分类器1350/20190411_res18_1349Model__iter_24000.caffemodel",
		outputblobs,
		"E:/globaldata/int8data", Size(112, 112), "trt1350_1024_tp_resnet18_1080ti.trtmodel");
}

string findmodel(const string& rootdir, const string& filter){

	PaVfiles vfs;
	paFindFiles(rootdir.c_str(), vfs, filter.c_str());

	if (vfs.empty()){
		errorExit("没有找到模型文件.\n");
	}

	if (vfs.size() > 1){
		for (auto& file : vfs)
			printf("%s\n", file.c_str());
		errorExit("找到多个模型.\n");
	}
	return vfs[0];
}

void generatorClassifier(const string& rootdir){
	std::vector<std::string> outputblobs
	{ "prob" };

	caffeToTRTModel(
		24,
		findmodel(rootdir, "*.prototxt"),
		findmodel(rootdir, "saved_*.caffemodel"),
		outputblobs,
		"E:/globaldata/int8data", Size(112, 112), rootdir + "/trt.trtmodel");
}

void main(int argc, char** argv){

	
	generatorTPRClassifier();
	return;

	if (argc < 2){
		printf("generator datadir\n");
		return;
	}

	const char* dir = argv[1];
	system(format("title %s", dir).c_str());
	printf("gen model: %s\n", dir);
	cudaSetDevice(0);
	initLibNvInferPlugins(&gLogger, "");

	generatorClassifier(dir);
	//generatorRefineDet();
	shutdownProtobufLibrary();
}