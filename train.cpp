


#include <cv.h>
#include <highgui.h>
#include "cc_nb.h"
#include <pa_file/pa_file.h>
#include "visualize.h"

using namespace cv;
using namespace cc;
using namespace std;

namespace L = cc::layers;

#define BatchSize		2
#define InputWidth		800
#define InputHeight		600

const int numclass = 1;
const int numpoint = 2;
const int connect[1][2] = { { 0, 1 } };
const string datapath = "";
const string suffix = "";




struct ObjPoint{
	float x, y;
	bool hidden;

	ObjPoint(float x, float y, const char* hidden) :
		x(x), y(y), hidden(strcmp(hidden, "True") == 0){}

	Point2f point() const{
		return Point2f(x, y);
	}
};

struct FileLine{
	string path;
	vector<ObjectPoints> objs;
};

struct ObjectPoints{
	int label;
	vector<ObjPoint> pts;
};

struct point_min_max_xy{
	float min_x;
	float min_y;
	float max_x;
	float max_y;
};



/*数据增广*/
int randr(int mi, int mx){
	if (mi > mx) std::swap(mi, mx);
	int r = mx - mi + 1;
	return rand() % r + mi;
}

float randr(float mi, float mx){
	float acc = rand() / (float)RAND_MAX;
	return acc * (mx - mi) + mi;
}

//1.随机翻转

/*
flipCode，翻转模式:
flipCode < 0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）,
flipCode == 0垂直翻转（沿X轴翻转），
flipCode > 0水平翻转（沿Y轴翻转）
*/
void augmentation_flip(Mat& img_src, vector<ObjectPoints>& item_pts){
	int mode;
	mode = 1;// randr(-1, 1);
	flip(img_src, img_src, mode);

	for (int i = 0; i < item_pts.size(); ++i){
		auto& item = item_pts[i].pts;
		if (mode == -1){
			for (int j = 0; j < item.size(); ++j){
				item[j].x = img_src.cols - item[j].x;
				item[j].y = img_src.rows - item[j].y;
			}
		}
		else if (mode == 0){ //x不变,y改变
			for (int j = 0; j < item.size(); ++j){
				item[j].y = img_src.rows - item[j].y;
			}
		}
		else if (mode == 1){//x改变，y不变
			for (int j = 0; j < item.size(); ++j){
				item[j].x = img_src.cols - item[j].x;
			}
		}
	}
}

//2.随机旋转

//template<typename Dtype>
void RotatePoint(vector<ObjPoint>& p, Mat R){

	for (int i = 0; i < p.size(); ++i){
		auto& item = p[i];
		Mat point(3, 1, CV_64FC1);
		point.at<double>(0, 0) = item.x;
		point.at<double>(1, 0) = item.y;
		point.at<double>(2, 0) = 1;
		Mat new_point = R * point;
		item.x = new_point.at<double>(0, 0);
		item.y = new_point.at<double>(1, 0);
	}
}

point_min_max_xy border_point(const vector<ObjectPoints>& item_pts){  //return 

	vector<float> vector_x;
	vector<float> vector_y;
	for (int i = 0; i < item_pts.size(); ++i){
		for (int j = 0; j < item_pts[i].pts.size(); ++j){
			vector_x.push_back(item_pts[i].pts[j].x);
			vector_y.push_back(item_pts[i].pts[j].y);
		}
	}

	struct point_min_max_xy point_;
	point_.min_x = *min_element(vector_x.begin(), vector_x.end());
	point_.min_y = *min_element(vector_y.begin(), vector_y.end());
	point_.max_x = *max_element(vector_x.begin(), vector_x.end());
	point_.max_y = *max_element(vector_y.begin(), vector_y.end());

	return point_;

}

void augmentation_rotate(Mat& img_src, vector<ObjectPoints>& item_pts){

	float angle;
	int n = randr(0, 1);
	if (n == 0){
		angle = randr(0.f, 30.f);
	}
	else	{
		angle = randr(330.f, 360.f);
	}


	float scale = 1;// randr(0.75f, 1.25f);

	Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
	Mat R = getRotationMatrix2D(center, angle, scale);

	Size size = img_src.size();
	Rect bbox = RotatedRect(center, Size(size.width*scale, size.height*scale), angle).boundingRect();
	// adjust transformation matrix
	R.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	R.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	warpAffine(img_src, img_src, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));
	for (int i = 0; i < item_pts.size(); ++i){
		RotatePoint(item_pts[i].pts, R);
	}
}

#if 0
//3.随机缩放
void augmentation_scale(Mat& img_src, vector<ObjectPoints>& item_pts){
	float scale = 1.25; randr(float(0.75), float(1.25));
	Mat img_scale;//缩放后的图片
	Mat img_roi;//背景图上截取的区域
	Mat img_background;//背景图

	int point_x, point_y; //point_x < img_src.cols - img_scale.cols
	//point_y < img_src.rows - img_scale.rows

	cv::resize(img_src, img_scale, Size(floor(img_src.cols * scale), floor(img_src.rows * scale)));

	point_x = randr(0, img_src.cols - img_scale.cols);
	point_y = randr(0, img_src.rows - img_scale.rows);


	img_background = img_src;
	img_background = Scalar::all(0);
	img_roi = img_background(Rect(point_x, point_y, img_scale.cols, img_scale.rows));
	img_scale.copyTo(img_roi);

	for (int i = 0; i < item_pts.size(); ++i){
		for (int j = 0; j < item_pts[i].pts.size(); ++j){
			item_pts[i].pts[j].x = floor(item_pts[i].pts[j].x * scale + point_x);
			item_pts[i].pts[j].y = floor(item_pts[i].pts[j].y * scale + point_y);
		}
	}
}
#endif

void augmentation_crop(Mat& img_src, vector<ObjectPoints>& item_pts){
	point_min_max_xy border_point_ = border_point(item_pts);
	int left_x_range = 0, left_y_range = 0, right_x_range = 0, right_y_range = 0;
	left_x_range = border_point_.min_x;
	left_y_range = border_point_.min_y;

	int pad = 30;
	int left_x, left_y, right_x, right_y;
	left_x = randr(0, max(0, left_x_range - pad));
	left_y = randr(0, max(0, left_y_range - pad));
	right_x = randr((float)min(border_point_.max_x + pad, img_src.cols - 1), (float)img_src.cols - 1);
	right_y = randr((float)min(border_point_.max_y + pad, img_src.rows - 1), (float)img_src.rows - 1);

	Mat img_crop = img_src(Rect(Point(left_x, left_y), Point(right_x, right_y)));
	img_src = img_crop;
	for (int i = 0; i < item_pts.size(); ++i){
		auto& item = item_pts[i].pts;
		for (int j = 0; j < item.size(); ++j){
			item[j].x = item[j].x - left_x;
			item[j].y = item[j].y - left_y;
		}
	}
}

//补全边界到与原图相同的比例(padding)
void padding_img(Mat& img_src, vector<ObjectPoints>& item_pts, int height, int width){

	int nw = img_src.cols;
	int nh = img_src.cols * height / (float)width;
	float nhacc = nh / (float)img_src.rows;
	int nh2 = img_src.rows;
	int nw2 = img_src.rows * width / (float)height;
	float nwacc = nw2 / (float)img_src.cols;
	int uw = 0, uh = 0;
	if (nhacc < nwacc && nh >= img_src.rows || nh >= img_src.rows && nw2 < img_src.cols){
		uw = nw;
		uh = nh;
	}
	else{
		uw = nw2;
		uh = nh2;
	}

	Mat background = Mat(uh, uw, img_src.type(), Scalar(128, 128, 128));
	int x = randr(0, uw - img_src.cols);
	int y = randr(0, uh - img_src.rows);
	img_src.copyTo(background(Rect(x, y, img_src.cols, img_src.rows)));

	for (int i = 0; i < item_pts.size(); ++i){
		auto& ps = item_pts[i].pts;
		for (auto& item : ps){
			item.x += x;
			item.y += y;
		}
	}
	background.copyTo(img_src);
}

void refreshPoint(FileLine& l){
	int p = l.path.rfind('.');
	string xml = l.path.substr(0, p) + ".txt";
	l.objs = loadPoints(xml, 0);
}



template<typename _Type>
_Type getGradientColor(const _Type& begin, const _Type& end, float val){
	return _Type(
		cv::saturate_cast<uchar>(begin[0] + (end[0] - begin[0])*val),
		cv::saturate_cast<uchar>(begin[1] + (end[1] - begin[1])*val),
		cv::saturate_cast<uchar>(begin[2] + (end[2] - begin[2])*val)
		);
}

void renderMask(Mat& rgbImage, const Mat& mask, Scalar color){

	CV_Assert(mask.size() == rgbImage.size());
	CV_Assert(CV_MAT_DEPTH(mask.type()) == CV_32F);

	for (int i = 0; i < mask.cols; ++i){
		for (int j = 0; j < mask.rows; ++j){
			float val = mask.at<float>(j, i);

			Vec3b& v = rgbImage.at<Vec3b>(j, i);
			v = getGradientColor(v, Vec3b(color[0], color[1], color[2]), val);
		}
	}
}

void splitString(const string& s, vector<string>& v, const string& c){
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
};


vector<ObjectPoints> loadPoints(const string& xml, bool* fail){
	vector<ObjectPoints> out;
	ifstream io;
	io.open(xml);
	if (!io.is_open())
		return out;

	while (!io.eof()){
		string buff;
		getline(io, buff);

		vector<string> vec_splitStr;
		splitString(buff, vec_splitStr, ",");
		if (vec_splitStr.size() < 38)
			continue;

		ObjectPoints ops;
		ops.label = getLabelmap().at("daoxian");
		vec_splitStr.erase(vec_splitStr.begin(), vec_splitStr.begin() + 2);
		for (int i = 0; i < vec_splitStr.size(); i += 3){
			Point tmpPoint;
			tmpPoint.x = atoi(vec_splitStr[i].c_str());
			tmpPoint.y = atoi(vec_splitStr[i + 1].c_str());
			string tmpStr;
			tmpStr = vec_splitStr[i + 2];
			if (i == 0 || i == vec_splitStr.size() - 3){
				ops.pts.emplace_back(ObjPoint(tmpPoint.x/1.6, tmpPoint.y/1.6, "False"));
			}
		}
		out.push_back(ops);
	}
	return out;
};

Mat loadMask(const string& file){

	int p = file.rfind('.');
	string path = file.substr(0, p) + "_mask.jpg";
	return imread(path, 0);
}

int randr(int low, int high){

	if (low > high) std::swap(low, high);
	return rand() % (high - low + 1) + low;
}

float randr(float low, float high){

	if (low > high) std::swap(low, high);
	return rand() / (float)RAND_MAX * (high - low) + low;
}

class Augment{

public:

	void augment(Mat& image, Mat& mask){

		this->image = image;
		this->mask = mask;

		int n = randr(0, 3);
		for (int i = 0; i < n; ++i){

			flip();
			rotate();
			adajust();
		}
	}

private:
	void flip(){

		if (randr(0, 1) == 0){

			int flag = randr(-1, 1);
			cv::flip(image, image, flag);
			cv::flip(mask, mask, flag);
		}
	}

	void rotate(){

		if (randr(0, 1) == 0){

			int cx = randr(0, image.cols);
			int cy = randr(0, image.rows);

			float angle = randr(0, 360);
			float scale = randr(0.8f, 1.2f);
			Mat m = getRotationMatrix2D(Point2f(cx, cy), angle, scale);
			cv::warpAffine(image, image, m, image.size(), INTER_AREA);		//INTER_NEAREST
			cv::warpAffine(mask, mask, m, image.size(), INTER_NEAREST);
		}
	}

	void adajust(){

		if (randr(0, 1) == 0){

			float light = randr(0.7f, 1.3f);
			image *= light;
		}
	}

private:
	Mat image, mask;
};

cc::Tensor myconv2d(cc::Tensor& input, int kernel_size, int num_output, const string& name, const float& k_lr_mult, const float& k_decay_mult, const float& b_lr_mult, const float& b_decay_mult){
	auto x = L::conv2d(input, { kernel_size, kernel_size, num_output }, "same", { 1, 1 }, { 1, 1 }, name);
	L::OConv2D* conv = (L::OConv2D*)x->owner.get();
	conv->bias_initializer.reset(new cc::Initializer());
	conv->kernel_initializer.reset(new cc::Initializer());

	conv->kernel_initializer->type = "gaussian";
	conv->kernel_initializer->stdval = 0.01;
	conv->bias_initializer->type = "constant";
	conv->bias_initializer->value = 0;

	conv->kernel_mult.reset(new cc::ParamSpecMult(k_lr_mult, k_decay_mult));
	conv->bias_mult.reset(new cc::ParamSpecMult(b_lr_mult, b_decay_mult));

	return x;
}


cc::Tensor myconv2d_iter(cc::Tensor& input, int kernel_size, int num_output, const float& k_lr_mult, const float& k_decay_mult, const float& b_lr_mult, const float& b_decay_mult, bool hasActive, int iter_num, int stage_layer, const String branch){
cc:Tensor x = input;
	if (hasActive){
		for (int i = 1; i <= iter_num; ++i){
			x = myconv2d(input, kernel_size, num_output, cc::sformat(("Mconv%d_stage%d_" + branch).c_str(), iter_num, stage_layer)
				, k_lr_mult, k_decay_mult, b_lr_mult, b_decay_mult);
			x = L::relu(x, cc::sformat(("Mrelu%d_stage%d" + branch).c_str(), iter_num, stage_layer));
		}
	}
	else{
		for (int i = 1; i <= iter_num; ++i){
			x = myconv2d(input, kernel_size, num_output, cc::sformat(("Mconv%d_stage%d_" + branch).c_str(), iter_num, stage_layer)
				, k_lr_mult, k_decay_mult, b_lr_mult, b_decay_mult);
		}
	}
	return x;
}


cc::Tensor myvgg(const cc::Tensor& input){
	cc::Tensor x = input;
	int numoutput = 64;
	for (int i = 1; i <= 3; ++i){
		int n = i <= 2 ? 2 : 4;
		for (int j = 1; j <= n; ++j){
			x = myconv2d(x, 3, numoutput * 2 ^ (i - 1), cc::sformat("conv%d_%d", i, j), 1.0f, 1.0f, 2.0f, 0.0f);
			x = L::relu(x, cc::sformat("relu%d_%d", i, j));
		}
		x = L::max_pooling2d(x, { 2, 2 }, { 2, 2 }, { 0, 0 }, false, cc::sformat("pool%d_stage1", i));
	}
	for (int k = 1; k <= 4; ++k){
		const string suf = k <= 2 ? "_CPM" : "";
		x = myconv2d(x, 3, 512, cc::sformat(("conv4_%d" + suf).c_str(), k), 1.0f, 1.0f, 2.0f, 0.0f);
		x = L::relu(x, cc::sformat(("relu4_%d" + suf).c_str(), k));
	}
	return x;
}


//stage1 的convnet
cc::Tensor conv_stage1(const cc::Tensor& input, bool is_L1){
	cc::Tensor x = input;
	int num_output = (is_L1) ? L1_numoutput : L2_numoutput;
	String branch = (is_L1) ? "_CPM_L1" : "_CPM_L2";
	for (int i = 1; i <= 3; ++i){
		x = myconv2d(x, 3, 128, cc::sformat(("conv5_%d" + branch).c_str(), i), 1.0f, 1.0f, 2.0f, 0.0f);
		x = L::relu(x, cc::sformat(("relu5_%d" + branch).c_str(), i));
	}
	x = myconv2d(x, 1, 512, "conv5_4" + branch, 1.0f, 1.0f, 2.0f, 0.0f);
	x = L::relu(x, "relu5_4" + branch);
	x = myconv2d(x, 1, num_output, "conv5_5" + branch, 1.0f, 1.0f, 2.0f, 0.0f);
	return x;
}


//后续stage的convnet
cc::Tensor Mconv_stage(const cc::Tensor& input, bool is_L1, int layer_stage){
	cc::Tensor x = input;
	int num_output = (is_L1) ? L1_numoutput : L2_numoutput;
	String branch = (is_L1) ? "L1" : "L2";
	x = myconv2d_iter(x, 7, 128, 4.0f, 1.0f, 8.0f, 0.0f, 1, 5, layer_stage, branch);
	x = myconv2d(x, 1, 128, cc::sformat(("Mconv6_stage%d_" + branch).c_str()), 4.0f, 1.0f, 8.0f, 0.0f);
	x = L::relu(x, cc::sformat(("Mrelu6_stage%d_" + branch).c_str(), layer_stage));
	x = myconv2d(x, 1, num_output, cc::sformat(("Mconv6_stage%d_" + branch)s.c_str(), layer_stage), 4.0f, 1.0f, 8.0f, 0.0f);
	return x;
}

void putGaussianMaps(float* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma){
	//LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
	float start = stride / 2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
	int uy = center.y / stride;
	int ux = center.x / stride;
	uy = uy >= grid_y ? grid_y - 1 : (uy < 0 ? 0 : uy);
	ux = ux >= grid_x ? grid_x - 1 : (ux < 0 ? 0 : ux);
	entry[uy * grid_x + ux] = 1;

	for (int g_y = 0; g_y < grid_y; g_y++){
		for (int g_x = 0; g_x < grid_x; g_x++){
			float x = start + g_x * stride;
			float y = start + g_y * stride;
			float d2 = (x - center.x)*(x - center.x) + (y - center.y)*(y - center.y);
			float exponent = d2 / 2.0 / sigma / sigma;
			if (exponent > 4.6052){ //ln(100) = -ln(1%)
				continue;
			}
			entry[g_y*grid_x + g_x] += exp(-exponent);
			if (entry[g_y*grid_x + g_x] > 1)
				entry[g_y*grid_x + g_x] = 1;
		}
	}
}

void putVecMaps(float* entryX, float* entryY, Mat& count, Point2f centerA, Point2f centerB,
	int stride, int grid_x, int grid_y, float thre){
	//int thre = 4;
	//centerB = centerB*0.125;
	//centerA = centerA*0.125;
	count.setTo(0);
	centerA = Point2f(centerA.x / (float)stride, centerA.y / (float)stride);
	centerB = Point2f(centerB.x / (float)stride, centerB.y / (float)stride);
	Point2f bc = centerB - centerA;
	int min_x = max(int(round(min(centerA.x, centerB.x) - thre)), 0);
	int max_x = min(int(round(max(centerA.x, centerB.x) + thre)), grid_x);

	int min_y = max(int(round(min(centerA.y, centerB.y) - thre)), 0);
	int max_y = min(int(round(max(centerA.y, centerB.y) + thre)), grid_y);

	float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
	bc.x = bc.x / norm_bc;
	bc.y = bc.y / norm_bc;

	// float x_p = (centerA.x + centerB.x) / 2;
	// float y_p = (centerA.y + centerB.y) / 2;
	// float angle = atan2f(centerB.y - centerA.y, centerB.x - centerA.x);
	// float sine = sinf(angle);
	// float cosine = cosf(angle);
	// float a_sqrt = (centerA.x - x_p) * (centerA.x - x_p) + (centerA.y - y_p) * (centerA.y - y_p);
	// float b_sqrt = 10; //fixed

	for (int g_y = min_y; g_y < max_y; g_y++){
		for (int g_x = min_x; g_x < max_x; g_x++){
			Point2f ba;
			ba.x = g_x - centerA.x;
			ba.y = g_y - centerA.y;
			float dist = std::abs(ba.x*bc.y - ba.y*bc.x);

			// float A = cosine * (g_x - x_p) + sine * (g_y - y_p);
			// float B = sine * (g_x - x_p) - cosine * (g_y - y_p);
			// float judge = A * A / a_sqrt + B * B / b_sqrt;

			if (dist <= thre){
				//if(judge <= 1){
				int cnt = count.at<uchar>(g_y, g_x);
				//LOG(INFO) << "putVecMaps here we start for " << g_x << " " << g_y;
				if (cnt == 0){
					entryX[g_y*grid_x + g_x] = bc.x;
					entryY[g_y*grid_x + g_x] = bc.y;
				}
				else{
					entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x] * cnt + bc.x) / (cnt + 1);
					entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x] * cnt + bc.y) / (cnt + 1);
					count.at<uchar>(g_y, g_x) = cnt + 1;
				}
			}
		}
	}
}

class InputData : public cc::BaseLayer{

public:
	SETUP_LAYERFUNC(InputData);

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop){
		const int batch_size = BatchSize;
		const int width = InputWidth;
		const int height = InputHeight;

		auto image = top[0];
		auto direction = top[1];
		auto keypoint = top[2];

		const int feature_map_width = width / 8;
			const int feature_map_height = height / 8;

		image->reshape(batch_size, 3, height, width);
		keypoint->reshape(batch_size, numclass * numpoint + 1, feature_map_height, feature_map_width);
		direction->reshape(batch_size, (sizeof(connect)/sizeof(connect[0])*numclass * 2, feature_map_height, feature_map_width);

		PaVfiles vfs;
		paFindFiles(datapath.c_str(), vfs, suffix.c_str(), false);
		bool fail = false; 
		for (int i = 0; i < vfs.size(); ++i){
			int p = vfs[i].rfind(".");
			string xml = vfs[i].substr(0, p) + ".txt";
			FileLine fl;
			fl.path = vfs[i];
			fl.objs = loadPoints(xml,&fail);

			if (fl.objs.size() > 0){
				fileLines_.emplace_back(fl);
			}
			else{
				printf("empty objs file: %s\n", fl.path.c_str());
			}

		};
		std::random_shuffle(this->fileLines_.begin(), this->fileLines_.end());
		cursor_ = 0;
	}

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop){


		auto image = top[0];
		auto direction = top[1];
		auto keypoint = top[2];

		Size imsize(image->width(), image->height());

		Mat dstmask;
		int batch = image->num();
		int width = image->width();
		int height = image->height();

		
		int feature_width = width / 8;
		int feature_height = height / 8;
		int feature_area = feature_width * feature_height;



		//Rect roi(0, 0, 1280, 960);

		//对原始图片做预处理
		for (int item_id = 0; item_id < batch_size; ++item_id){
			auto item = this->fileLines_[this->cursor_];
			refreshPoint(item);

			while (item.objs.size() == 0){
				printf("empty objs file: %s\n", item.path.c_str());
				this->cursor_++;
				if (this->cursor_ == this->fileLines_.size()){
					this->cursor_ = 0;
					std::random_shuffle(this->fileLines_.begin(), this->fileLines_.end());
				}

				item = this->fileLines_[this->cursor_];
				refreshPoint(item);
			}
			int padding_height = 600;
			int padding_width = 800;
			Mat im = imread(item.path);


			//增广
			int aug_num; //增广次数
			aug_num = randr(1, 3);
			//aug_num = 1;
			int rand_aug;
			vector<int> aug_inds;
			for (int i = 0; i < 3; ++i)
				aug_inds.push_back(i);

			std::random_shuffle(aug_inds.begin(), aug_inds.end());
			//cv::medianBlur(im, im, 3);

			for (int i = 0; i < aug_num; ++i){
				rand_aug = aug_inds[i];
				if (rand_aug == 0){
					augmentation_flip(im, item.objs);
				}
				else if (rand_aug == 1){
					augmentation_rotate(im, item.objs);
					//augmentation_crop(im, item.objs);
					padding_img(im, item.objs, padding_height, padding_width);
				}
				//else if (rand_aug == 2){
				//	augmentation_crop(im, item.objs);
				//	padding_img(im, item.objs, padding_height, padding_width);
				//}
			}

			float scale_x = width / (float)im.cols;
			float scale_y = height / (float)im.rows;

#if 0
			for (int k = 0; k < item.objs.size(); ++k){
				for (int m = 0; m < item.objs[k].pts.size(); ++m){
					circle(im, item.objs[k].pts[m].point(), 5, Scalar(0, 255), -1);
				}
			}
#endif

			cv::resize(im, im, Size(width, height));
			im.convertTo(im, CV_32F, 1 / 255.f, -0.5);
			
			image->setData(item_id, im);


			for (int i = 0; i < item.objs.size(); ++i){
				for (int j = 0; j < item.objs[i].pts.size(); ++j){
					if (!item.objs[i].pts[j].hidden){
						putGaussianMaps(heat_temp_ptr + ((item.objs[i].label - 1) * config.numpoint + j + 1) * feature_area,
							Point2f(item.objs[i].pts[j].x * scale_x, item.objs[i].pts[j].y * scale_y), 8, feature_width, feature_height,
							5);
						//item.objs[i].label >= 6 && item.objs[i].label <= 8 ? feature_height * 1/4.0 : 5.0);
					}\
				}
			}
			Mat count = Mat::zeros(feature_height, feature_width, CV_8U);

			for (int o = 0; o < item.objs.size(); ++o){
				int numConnect = config.connect.size();
				int label_off = (item.objs[o].label - 1) * (numConnect * 2)* feature_area;
				//int label_off = 0;
				float thre = 1;

				for (int j = 0; j < numConnect; ++j){
					int indA = config.connect[j].first;
					int indB = config.connect[j].second;
					if (!item.objs[o].pts[indA].hidden && !item.objs[o].pts[indB].hidden){
						putVecMaps(
							vec_temp_ptr + label_off + (j * 2 + 0) * feature_area,
							vec_temp_ptr + label_off + (j * 2 + 1) * feature_area,
							//vec_temp_ptr + (j * 2 + 0) * feature_area,
							//vec_temp_ptr + (j * 2 + 1) * feature_area,
							count,
							Point2f(item.objs[o].pts[indA].x*scale_x, item.objs[o].pts[indA].y*scale_y),
							Point2f(item.objs[o].pts[indB].x*scale_x, item.objs[o].pts[indB].y*scale_y),
							8, feature_width, feature_height, thre);
					}
				}
			}

			float* heat0_ptr = heat_temp_ptr + 0 * feature_area;
			for (int y = 0; y < feature_height; ++y){
				for (int x = 0; x < feature_width; ++x){
					heat0_ptr[x] = 0;
					for (int c = 1; c < heat_temp->channel(); ++c){
						float val = *(heat_temp_ptr + c * feature_area + y * feature_width + x);
						heat0_ptr[x] = max(val, heat0_ptr[x]);
					}
					heat0_ptr[x] = 1 - heat0_ptr[x];
				}
				heat0_ptr += feature_width;
			}

			heat_temp_ptr += heat_temp->channel() * feature_area;
			vec_temp_ptr += vec_temp->channel() * feature_area;

			this->cursor_++;
			if (this->cursor_ == this->fileLines_.size()){
				this->cursor_ = 0;
				std::random_shuffle(this->fileLines_.begin(), this->fileLines_.end());
			}




			/*for (int i = 0; i < batch; ++i){
				Mat image = imread(files_[cursor_]);
				Mat mask = loadMask(files_[cursor_]);

				cv::dilate(mask, mask, getStructuringElement(CV_SHAPE_RECT, Size(5, 5)));
				cv::GaussianBlur(mask, mask, Size(3, 3), 3.0);
				resize(image, image, imsize);
				resize(mask, mask, imsize);

				mask.convertTo(mask, CV_32F);
				mask.setTo(0, mask <= 100);
				mask.setTo(1, mask > 100);
				augment_.augment(image, mask);

				image.convertTo(image, CV_32F, 1 / 255.0, -0.5);
				image.copyTo(image);

				merge({ 1 - mask, mask }, dstmask);
				blob_image->setData(i, image);
				blob_mask->setData(i, dstmask);

				if (++cursor_ == files_.size()){
				std::random_shuffle(files_.begin(), files_.end());
				cursor_ = 0;
				}
				*/}
	}

private:
	vector<FileLine> fileLines_;
	Augment augment_;
	int cursor_ = 0;
};

cc::Tensor vggconv2d(cc::Tensor& input, int kernel_size, int num_output, const string& name){
	auto x = L::conv2d(input, { kernel_size, kernel_size, num_output }, "same", { 1, 1 }, { 1, 1 }, name);
	L::OConv2D* conv = (L::OConv2D*)x->owner.get();
	conv->bias_initializer.reset(new cc::Initializer());
	conv->kernel_initializer.reset(new cc::Initializer());

	conv->kernel_initializer->type = "gaussian";
	conv->kernel_initializer->stdval = 0.01;
	conv->bias_initializer->type = "constant";
	conv->bias_initializer->value = 0;
	return x;
}

cc::Tensor vgg(const cc::Tensor& input){

	cc::Tensor x = input;
	int num_output = 64;

	int numblock[] = { 0, 2, 2, 4, 2 };
	for (int i = 1; i <= 4; ++i){

		int n = numblock[i];
		for (int j = 1; j <= n; ++j){
			x = vggconv2d(x, 3, num_output, cc::sformat("conv%d_%d", i, j));
			x = L::relu(x, cc::sformat("relu%d_%d", i, j));
		}

		if (i < 4)
			x = L::max_pooling2d(x, { 2, 2 }, { 2, 2 }, { 0, 0 }, false, cc::sformat("pool%d_stage1", i));

		if (i < 4)
			num_output *= 2;
	}
	return x;
}

cc::Tensor pathLine(const cc::Tensor& input,
	const string& convnamefmt, const string& relunamefmt, const vector<int>& numoutput, int kernelSize){

	auto x = input;
	for (int i = 1; i <= numoutput.size(); ++i){

		int usekernelsize;
		if (i <= numoutput.size() - 2)
			usekernelsize = kernelSize;
		else
			usekernelsize = 1;

		x = vggconv2d(x, usekernelsize, numoutput[i - 1], cc::sformat(convnamefmt.c_str(), i));

		if (i != numoutput.size())
			x = L::relu(x, cc::sformat(relunamefmt.c_str(), i));
	}
	return x;
}

cc::Tensor pose(const cc::Tensor& input, int num_stage, int l1Output = 38, int l2Output = 19){

	auto x = vgg(input);
	x = vggconv2d(x, 3, 256, cc::sformat("conv%d_%d_CPM", 4, 3));
	x = L::relu(x, cc::sformat("relu%d_%d_CPM", 4, 3));

	x = vggconv2d(x, 3, 128, cc::sformat("conv%d_%d_CPM", 4, 4));
	x = L::relu(x, cc::sformat("relu%d_%d_CPM", 4, 4));

	auto backbone = x;
	auto l1 = pathLine(backbone, "conv5_%d_CPM_L1", "relu5_%d_CPM_L1", { 128, 128, 128, 512, l1Output }, 3);
	auto l2 = pathLine(backbone, "conv5_%d_CPM_L2", "relu5_%d_CPM_L2", { 128, 128, 128, 512, l2Output }, 3);
	x = L::concat({ backbone, l1, l2 }, 1, "concat_stage2");

	for (int i = 0; i < num_stage; ++i){

		int stage = i + 2;
		auto l1 = pathLine(x, cc::sformat("Mconv%%d_stage%d_L1", stage), cc::sformat("Mrelu%%d_stage%d_L1", stage), { 128, 128, 128, 128, 128, 128, l1Output }, 7);
		auto l2 = pathLine(x, cc::sformat("Mconv%%d_stage%d_L2", stage), cc::sformat("Mrelu%%d_stage%d_L2", stage), { 128, 128, 128, 128, 128, 128, l2Output }, 7);

		if (i < num_stage - 1)
			x = L::concat({ backbone, l1, l2 }, 1, cc::sformat("concat_stage%d", stage + 1));
		else
			x = L::concat({ l1, l2 }, 1, cc::sformat("concat_stage%d", stage + 1));
	}
	return x;
}

cc::Tensor pose2(const cc::Tensor& input, int num_stage, int l2Output = 19){

	auto x = vgg(input);
	x = vggconv2d(x, 3, 256, cc::sformat("conv%d_%d_CPM", 4, 3));
	x = L::relu(x, cc::sformat("relu%d_%d_CPM", 4, 3));

	x = vggconv2d(x, 3, 128, cc::sformat("conv%d_%d_CPM", 4, 4));
	x = L::relu(x, cc::sformat("relu%d_%d_CPM", 4, 4));

	auto backbone = x;
	auto l2 = pathLine(backbone, "conv5_%d_CPM_L2", "relu5_%d_CPM_L2", { 128, 128, 128, 512, l2Output }, 3);
	x = L::concat({ backbone, l2 }, 1, "concat_stage2");

	for (int i = 0; i < num_stage; ++i){

		int stage = i + 2;
		auto l2 = pathLine(x, cc::sformat("Mconv%%d_stage%d_L2", stage), cc::sformat("Mrelu%%d_stage%d_L2", stage), { 128, 128, 128, 128, 128, 128, l2Output }, 7);

		if (i < num_stage - 1)
			x = L::concat({ backbone, l2 }, 1, cc::sformat("concat_stage%d", stage + 1));
		else
			x = l2;
	}
	return x;
}

void trainStep(OThreadContextSession* session, int step, float smoothed_loss){

	Blob* predict = session->get_tensor_blob("Mconv7_stage2_L2");
	Blob* mask = session->get_tensor_blob("mask");
	Blob* image = session->get_tensor_blob("image");

	postBlob(predict, "predict-norm");
	postBlob(image, "image");
	postBlob(mask, "mask", 0, 0, 0, norNorm);
	postBlob(predict, "predict", 0, 0, 0, norNorm);
}

int main(){

	cc::installRegister();
	INSTALL_LAYER(InputData);
	initializeVisualze();
	setGPU(0);
	{
		auto datatrain = L::data("InputData", { "image", "mask" }, "input");
		auto image = datatrain[0];
		auto mask = datatrain[1];

		auto predict = pose2(image, 1, 2);
		auto loss = cc::loss::euclidean(predict, mask, nullptr, "mask_loss");
		loss->owner->propagate_down = { 1, 0 };
		loss->owner->loss_weight = 1.0f;

		auto op = cc::optimizer::momentumStochasticGradientDescent(cc::learningrate::step(1e-6, 0.1, 10000), 0.9);
		op->weight_decay = 0.0002f;
		op->average_loss = 1;
		op->max_iter = 99999999;
		op->display = 10;
		op->snapshot = 5000;
		op->device_ids = { 0 };
		op->snapshot_prefix = "models/unet";
		op->reload_weights = "pose_iter_440000.caffemodel";
		op->minimize({ loss });
		cc::engine::caffe::buildGraphToFile({ loss }, "train.prototxt");

		//保存inference模型，用来查看
		{
			auto image_inference = L::input({ 1, 3, InputHeight, InputWidth }, "image");
			auto predict_inference = pose2(image_inference, 1, 2);
			cc::engine::caffe::buildGraphToFile({ predict_inference }, "inference.prototxt");
		};

		cc::train::caffe::run(op, trainStep, [](OThreadContextSession* session){
			setSolver(session->solver());
		});
	};
	destoryVisualze();
	return 0;
}