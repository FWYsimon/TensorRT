

#ifdef DTest
#include "tensorRTEngine.hpp"
#include <pa_file\pa_file.h>

#pragma comment(lib, "libcaffe.lib")


vector<ObjectInfo> toDetectionObjs(Blob* fr, float threshold, int imWidth = 1, int imHeight = 1) {
	vector<ObjectInfo> out;
	float* data = fr->mutable_cpu_data();
	for (int i = 0; i < fr->count(); i += 7, data += 7) {
		ObjectInfo obj;

		//if invalid det
		if (data[0] == -1 || data[2] < threshold /*|| data[1] ==2*/)
			continue;

		obj.image_id = data[0];
		obj.label = data[1];
		obj.score = data[2];
		obj.xmin = data[3] * imWidth;
		obj.ymin = data[4] * imHeight;
		obj.xmax = data[5] * imWidth;
		obj.ymax = data[6] * imHeight;
		out.push_back(obj);
	}
	return out;
}

float areaFor(const ObjectInfo& a) {
	return (a.xmax - a.xmin) * (a.ymax - a.ymin);
}

const static int iou_min = 0;
const static int iou_union = 1;
float IoU(const ObjectInfo& a, const ObjectInfo& b, int type) {
	float xmax = max(a.xmin, b.xmin);
	float ymax = max(a.ymin, b.ymin);
	float xmin = min(a.xmax, b.xmax);
	float ymin = min(a.ymax, b.ymax);
	//Union

	float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
	float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
	float iou = uw * uh;

	if (type == iou_min)
		return iou / min(areaFor(a), areaFor(b));
	else
		return iou / (areaFor(a) + areaFor(b) - iou);
}

vector<ObjectInfo> nms(vector<ObjectInfo>& objs, float nmsThreshold_union) {
	std::sort(objs.begin(), objs.end(), [](const ObjectInfo& a, const ObjectInfo& b) {
		return a.score > b.score;
	});

	vector<ObjectInfo> out;
	vector<int> flags(objs.size());
	for (int i = 0; i < objs.size(); ++i) {
		if (flags[i] == 1) continue;

		out.push_back(objs[i]);
		flags[i] = 1;
		for (int k = i + 1; k < objs.size(); ++k) {
			if (flags[k] == 0) {
				float iouUnion = IoU(objs[i], objs[k], iou_union);
				if (iouUnion > nmsThreshold_union)
					flags[k] = 1;
			}
		}
	}
	return out;
}

void test_classifier(){
	const char* model = "分类器测试/9494_b2_classifier_p4000.trtmodel";
	InferenceEngine* trt = createTensorRTInt8InferenceEngine();

	if (!trt->loadFromFile(model)){
		printf("load model fail: %s\n", model);
		return;
	}

	PaVfiles vfs;
	paFindFiles("E:/样本收集/大兰测试模型/图片/标准测试数据/整理level1/小图", vfs, "*.jpg");

	for (auto& file : vfs){
		Mat im = imread(file);
		resize(im, im, Size(94, 94));
		cvtColor(im, im, CV_BGR2RGB);

		im.convertTo(im, CV_32F, 1 / 255.0);
		trt->inference(im);

		Blob* output = trt->outputBlob(0);
		string featpath = file.substr(0, file.rfind('.')) + ".feat";
		paWriteToFile(featpath.c_str(), output->cpu_data(), output->count()*sizeof(float));
		printf("%s\n", file.c_str());
	}
}

void test_detection(){
	initTensorRT();
	const char* model = "ref512_b2_p4000.trtmodel";
	TensorRTDetection* trt = createTensorRTDetection();

	if (!trt->loadFromFile(model)){
		printf("load model fail: %s\n", model);
		return;
	}

	VideoCapture cap("E:/样本收集/大兰-11楼样本/测试视频/0.avi");
	Mat frame, show;

	while (cap.read(frame)){

		frame.copyTo(show);

		double tick = getTickCount();
		Blob* output = trt->inference(frame);
		auto objs = toDetectionObjs(output, 0.5, frame.cols, frame.rows);
		tick = (getTickCount() - tick) / getTickFrequency() * 1000;

		printf("time: %.2f ms\n", tick);
		objs = nms(objs, 0.5);
		for (auto& obj : objs){
			rectangle(show, obj.box(), Scalar(0, 255), 2);
		}

		imshow("show", show);
		waitKey(30);
	}
}

void main(){
	cc::setGPU(1);
	test_detection();
}
#endif