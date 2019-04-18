


#include "tensorRTEngine.hpp"
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
#include <pa_file\pa_file.h>

#pragma comment(lib, "libcaffe.lib")
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
using namespace cc;

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
	{}

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

class TRTModel{

public:
	TRTModel(){};
	virtual ~TRTModel(){ release(); }

	bool loadFromFile(const string& model){
		release();

		size_t size;
		uchar* data = paReadFile(model.c_str(), &size);
		if (!data) return false;

		bool ok = loadFromData(data, size);
		freeReadFile(&data);
		return ok;
	}

	bool loadFromData(const void* model, int length){
		release();
		if (!model || length < 1) return false;

		runtime_ = createInferRuntime(gLogger);
		engine_ = runtime_->deserializeCudaEngine(model, length, nullptr);
		context_ = engine_->createExecutionContext();

		int maxBatchSize = engine_->getMaxBatchSize();
		int numBinding = engine_->getNbBindings();
		for (int i = 0; i < numBinding; ++i){
			if (engine_->bindingIsInput(i)){
				nvinfer1::Dims dims = engine_->getBindingDimensions(i);
				inputBlob_ = newBlobByShape(maxBatchSize, dims.d[0], dims.d[1], dims.d[2]);
				allblobs_.push_back(inputBlob_);
			}
			else{
				nvinfer1::Dims dims = engine_->getBindingDimensions(i);
				WPtr<Blob> blob = newBlobByShape(maxBatchSize, dims.d[0], dims.d[1], dims.d[2]);
				allblobs_.push_back(blob);
				outputBlobs_.push_back(blob);
			}
			blobMap_[engine_->getBindingName(i)] = i;
		}
		return true;
	}

#define freeInfr(p)		{if(p){p->destroy(); p = nullptr;}};
#define freeBlob(b)		{if(b){b.releaseRef();}}

	void release(){
		freeInfr(engine_);
		freeInfr(context_);
		freeInfr(runtime_);
		freeBlob(inputBlob_);
		allblobs_.clear();
	}

	int getMaxBatchSize(){
		if (engine_)
			return engine_->getMaxBatchSize();
		return 0;
	}

	void setToInputBlob(const vector<Mat>& frames){

		Size inputsize(inputBlob_->width(), inputBlob_->height());
		inputBlob_->Reshape(frames.size(), -1, -1, -1);

		for (int i = 0; i < outputBlobs_.size(); ++i)
			outputBlobs_[i]->Reshape(frames.size(), -1, -1, -1);

		for (int i = 0; i < frames.size(); ++i)
			inputBlob_->setDataRGB(i, frames[i]);
	}

	void** getBindings(){
		bindings_.resize(allblobs_.size());
		
		for (int i = 0; i < bindings_.size(); ++i)
			bindings_[i] = allblobs_[i]->mutable_gpu_data();
		return bindings_.data();
	}

	void inference(const vector<Mat>& frames){
		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));

		setToInputBlob(frames);
		if (!context_->enqueue(frames.size(), getBindings(), stream, nullptr)){
			printf("enqueue fail.\n");
		}
		//if (!context_->execute(frames.size(), getBindings())){
		//	printf("execute fail.\n");
		//}
		CHECK(cudaStreamSynchronize(stream));
		CHECK(cudaStreamDestroy(stream));
	}

	void inference(const Mat& frame){
		inference(vector<Mat>{ frame });
	}

	vector<WPtr<cc::Blob>>& getOutputBlobs(){
		return this->outputBlobs_;
	}

	vector<WPtr<cc::Blob>>& getAllBlobs(){
		return this->allblobs_;
	}

	WPtr<cc::Blob>& getInputBlob(){
		return this->inputBlob_;
	}

	cc::Blob* blob(const string& name){
		if (this->blobMap_.find(name) == this->blobMap_.end())
			return nullptr;

		return allblobs_[this->blobMap_[name]].get();
	}

	Size inputSize(){
		return Size(inputBlob_->width(), inputBlob_->height());
	}

private:
	ICudaEngine* engine_ = nullptr;
	IExecutionContext* context_ = nullptr;
	IRuntime* runtime_ = nullptr;
	WPtr<cc::Blob> inputBlob_ = nullptr;
	vector<WPtr<cc::Blob>> allblobs_;
	vector<WPtr<cc::Blob>> outputBlobs_;
	vector<void*> bindings_;
	map<string, int> blobMap_;
};

class RefineDet : public TensorRTDetection{
public:
	RefineDet(){}
	virtual int maxBatchSize(){
		return trt_.getMaxBatchSize();
	}

	virtual ~RefineDet(){
		release();
	}

	virtual bool loadFromFile(const char* model, const char* layerDefine){
		if (!trt_.loadFromFile(model))
			return false;

		setup(layerDefine);
		return true;
	}

	virtual bool loadFromData(const void* model, int length, const char* layerDefine){
		if (!trt_.loadFromData(model, length))
			return false;

		setup(layerDefine);
		return true;
	}

	void release(){
		detectout_.releaseRef();
		trt_.release();
	}

	Blob* inference(int numFrames, const Mat* frames){

		if (numFrames > this->maxBatchSize()){
			printf("错误，必须在要求的batchsize范围以内工作， numFrames[%d] > maxBatchSize[%d]\n", numFrames, this->maxBatchSize());
			return nullptr;
		}

		if (!detectout_.get()) return nullptr;

		Size isize = inputSize();
		vector<Mat> inputMats(numFrames);
		for (int i = 0; i < numFrames; ++i){

			Mat& inputmat = inputMats[i];
			if (frames[i].size() != isize)
				resize(frames[i], inputmat, isize);
			else
				frames[i].copyTo(inputmat);

			if (CV_MAT_DEPTH(inputmat.type()) != CV_32F)
				inputmat.convertTo(inputmat, CV_32F);

			inputmat -= Scalar(104.0, 117.0, 123.0);
		}
		trt_.inference(inputMats);

		Blob* bottom[] = {
			trt_.blob("odm_loc"), trt_.blob("odm_conf_flatten"),
			trt_.blob("arm_priorbox"), trt_.blob("arm_conf_flatten"), trt_.blob("arm_loc") };
		Blob* top[] = { detection_out_.get() };

		int numbottom = sizeof(bottom) / sizeof(bottom[0]);
		int numtop = sizeof(top) / sizeof(top[0]);
		for (int i = 0; i < numbottom; ++i)
			bottom[i]->Reshape(numFrames, -1, -1, -1);

		detectout_->forward((const Blob**)bottom, numbottom, (const Blob**)top, numtop);
		detection_out_->mutable_cpu_data();
		return detection_out_.get();
	}

	Size inputSize(){
		return trt_.inputSize();
	}

	Blob* inference(const Mat& frame){
		return inference(1, &frame);
	}

private:
	void setup(const char* layerDefine = nullptr){
		loadDetectoutLayer(layerDefine);
	}

	void loadDetectoutLayer(const char* layerDefine = nullptr){

		const char* defaultDefine =
			"  name: \"detection_out\"				   "
			"  type: \"DetectionOutput\"			   "
			"  bottom: \"odm_loc\"					   "
			"  bottom: \"odm_conf_flatten\"			   "
			"  bottom: \"arm_priorbox\"				   "
			"  bottom: \"arm_conf_flatten\"			   "
			"  bottom: \"arm_loc\"					   "
			"  top: \"detection_out\"				   "
			"  include {							   "
			"	phase: TEST							   "
			"  }									   "
			"  detection_output_param {				   "
			"	num_classes: 81						   "
			"	share_location: true				   "
			"	background_label_id: 0				   "
			"	nms_param {							   "
			"	  nms_threshold: 0.7				   "
			"	  top_k: 100      					   "
			"	}									   "
			"	code_type: CENTER_SIZE				   "
			"	keep_top_k: 100						   "
			"	confidence_threshold: 0.1			   "
			"	objectness_score: 0.00999999977648	   "
			"  }									   ";

		detectout_ = cc::newLayer(layerDefine == nullptr ? defaultDefine : layerDefine);
		
		vector<Blob*> bottomBlobs(detectout_->getNumBottom());
		for (int i = 0; i < bottomBlobs.size(); ++i)
			bottomBlobs[i] = trt_.blob(detectout_->bottomName(i).c_str());

		detection_out_ = newBlob();
		Blob* top[] = { detection_out_.get() };
		Blob** bottom = bottomBlobs.data();
		int numBottom = bottomBlobs.size();
		detectout_->setup((const Blob**)bottom, numBottom, (const Blob**)top, sizeof(top) / sizeof(top[0]));
	}

private:

	TRTModel trt_;
	WPtr<Layer> detectout_;
	WPtr<Blob> detection_out_;
};

void DETECTIONCALL initTensorRT(){
	static volatile bool inited = false;
	if (inited) return;
	inited = true;
	initLibNvInferPlugins(&gLogger, "");
}

TensorRTDetection* DETECTIONCALL createTensorRTDetection(){
	return new RefineDet();
}

void DETECTIONCALL releaseTensorRTDetection(TensorRTDetection* ptr){
	if (ptr) delete ptr;
}




////////////////////////////////////////////////
class TensorRTEngineImpl : public InferenceEngine{
public:
	TensorRTEngineImpl(){}
	virtual int maxBatchSize(){
		return trt_.getMaxBatchSize();
	}

	virtual ~TensorRTEngineImpl(){
		release();
	}

	virtual bool loadFromFile(const char* model){
		return trt_.loadFromFile(model);
	}

	virtual bool loadFromData(const void* model, int length){
		return trt_.loadFromData(model, length);
	}

	void release(){
		trt_.release();
	}

	Blob* inference(int numFrames, const Mat* frames){

		if (numFrames > this->maxBatchSize()){
			printf("错误，必须在要求的batchsize范围以内工作， numFrames[%d] > maxBatchSize[%d]\n", numFrames, this->maxBatchSize());
			return nullptr;
		}

		trt_.inference(vector<Mat>(frames, frames + numFrames));
		if (countOutput() == 1)
			return outputBlob(0);
		return nullptr;
	}

	//获取输出的个数
	int countOutput(){
		return trt_.getOutputBlobs().size();
	}

	Blob* blob(const char* blobName){
		return trt_.blob(blobName);
	}

	Size inputSize(){
		return trt_.inputSize();
	}

	Blob* inference(const Mat& frame){
		return inference(1, &frame);
	}

	Blob* outputBlob(int index){
		if (trt_.getOutputBlobs().empty()) 
			return nullptr;

		return trt_.getOutputBlobs()[0];
	}

private:
	TRTModel trt_;
};

InferenceEngine* DETECTIONCALL createTensorRTInt8InferenceEngine(){
	return new TensorRTEngineImpl();
}

void DETECTIONCALL releaseTensorRTInt8InferenceEngine(InferenceEngine* ptr){
	if (ptr) delete ptr;
}