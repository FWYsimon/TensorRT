

#pragma once

#include <cv.h>
#include <cc_utils.h>
using namespace cc;
using namespace cv;

#ifdef EXPORT_TO_DLL
#define DETECTIONAPI __declspec(dllexport)  
#else
#define DETECTIONAPI __declspec(dllimport)  
#endif

#define DETECTIONCALL __stdcall

//tensorRT int8 detection
class DETECTIONAPI TensorRTDetection{
public:
	virtual bool loadFromFile(const char* model, const char* layerDefine = 0) = 0;
	virtual bool loadFromData(const void* model, int length, const char* layerDefine = 0) = 0;
	virtual Blob* inference(int numFrames, const Mat* frames) = 0;
	virtual Blob* inference(const Mat& frame) = 0;
	virtual Size inputSize() = 0;
	virtual int maxBatchSize() = 0;
};

DETECTIONAPI void DETECTIONCALL initTensorRT();
DETECTIONAPI TensorRTDetection* DETECTIONCALL createTensorRTDetection();
DETECTIONAPI void DETECTIONCALL releaseTensorRTDetection(TensorRTDetection* ptr);


//tensorRT int8 inference
class DETECTIONAPI InferenceEngine{
public:
	virtual bool loadFromFile(const char* model) = 0;
	virtual bool loadFromData(const void* model, int length) = 0;

	//����ж��������򷵻�nullptr������е���������򷵻�ָ��
	virtual Blob* inference(int numFrames, const Mat* frames) = 0;
	virtual Blob* inference(const Mat& frame) = 0;
	virtual Size inputSize() = 0;
	virtual int maxBatchSize() = 0;
	virtual Blob* blob(const char* blobName) = 0;

	//��ȡ�����blob
	virtual Blob* outputBlob(int index) = 0;
	virtual int countOutput() = 0;
};

DETECTIONAPI InferenceEngine* DETECTIONCALL createTensorRTInt8InferenceEngine();
DETECTIONAPI void DETECTIONCALL releaseTensorRTInt8InferenceEngine(InferenceEngine* ptr);