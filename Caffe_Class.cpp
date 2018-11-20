#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <opencv2/videoio.hpp>

using namespace std;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}
static std::vector<String> readClassNames(const char *filename = "Ball.txt")
{
    std::vector<String> classNames;
    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }
    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }
    fp.close();
    return classNames;
}
const char* params
    = "{ help           | false | Sample app for loading googlenet model }"
      "{ proto          | bvlc_googlenet.prototxt | model configuration }"
      "{ model          | bvlc_googlenet.caffemodel | model weights }"
      "{ image          | space_shuttle.jpg | path to image file }"
      "{ opencl         | false | enable OpenCL }"
;
//int main(int argc, char **argv)
int main(int argc, char const *argv[])
{
	VideoCapture cap("rtsp://192.168.1.10/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1");	
	//VideoCapture cap(0);//Open dfault camera

	if(cap.isOpened()==false)
	{
		cout<<"Cannot open camera"<<endl;
		cin.get();
		return -1;
	}

	Mat OrgVid;

    CV_TRACE_FUNCTION();
    CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    while(true)
	{
		bool bSuccess = cap.read(OrgVid);	
		if(bSuccess==false)
		{
			cout<<"Video Camera was disconnected"<<endl;
			cin.get();
			break;
		}	

		imwrite("/home/agniv/Trial_Caffe/check.jpg",OrgVid);

    	//String modelTxt = parser.get<string>("proto");
    	String modelTxt = "/home/agniv/Trial_Caffe/deploy1.prototxt";
    	//String modelBin = parser.get<string>("model");
    	String modelBin = "/home/agniv/Trial_Caffe/snapshot_iter_1230.caffemodel";
    	//String imageFile = parser.get<String>("image");
    	String imageFile = "/home/agniv/Trial_Caffe/check.jpg";

    	Net net;
    	
    	net = dnn::readNetFromCaffe(modelTxt, modelBin);
    	 
    	
    	Mat img = imread(imageFile);


    	if (img.empty())
  		{
      	  std::cerr << "Can't read image from the file: " << imageFile << std::endl;
      	  exit(-1);
    	}	
    	//GoogLeNet accepts only 224x224 BGR-images
    	Mat inputBlob = blobFromImage(img, 1.0f, Size(227, 227),
    	                              Scalar(104, 117, 123), false);   //Convert Mat to batch of images
    	net.setInput(inputBlob, "data");        //set the network input
    	Mat softmax = net.forward("softmax");         //compute output
    	cv::TickMeter t;
    	for (int i = 0; i < 10; i++)
    	{
    	    CV_TRACE_REGION("forward");
    	    net.setInput(inputBlob, "data");        //set the network input
    	    t.start();
    	    softmax = net.forward("softmax");                          //compute output
    	    t.stop();
    	}
    	int classId;
    	double classProb;
   		getMaxClass(softmax, &classId, &classProb);//find the best class
    	std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
		switch(classId)
		{
			case 0:
				if(classProb>=0.9)
					cout<<"Ball"<<endl;
				break;
			case 1:
				cout<<"No Ball"<<endl;
				break;
		}

		namedWindow("OrgVid", WINDOW_KEEPRATIO);
    	imshow("OrgVid",OrgVid);

    	if(waitKey(10)==27)
		{
			cout<<"Exit"<<endl;
			break;
		}
    }
    return 0;
} //main
