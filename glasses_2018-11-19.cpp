/**
 * @file objectDetection2.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to
 *       load a cascade classifier and how to find objects (Face + eyes)
 *       in a video stream - Using LBP here
 */
// kod zmodyfikowany na potrzeby zajęć z NAI na PJATK Gdańsk

// g++ -fopenmp `pkg-config --cflags opencv` glasses.cpp -o glasses `pkg-config --libs opencv`
// okulary pobrane z http://memesvault.com/wp-content/uploads/Deal-With-It-Sunglasses-07.png
// elementy związane z przekształceniem geometrycznym http://dsynflo.blogspot.in/2014/08/simplar-2-99-lines-of-code-for.html
// zachęcam do zapoznania się z https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

// kapelusz pobrany z http://www.pngpix.com/wp-content/uploads/2016/07/PNGPIX-COM-Cowboy-Hat-PNG-Transparent-Image-2-2-500x349.png

// chusta pobrana z
// https://banner2.kisspng.com/20180907/ukg/kisspng-diving-snorkeling-masks-balaclava-scarf-face--5b929695aece07.364719161536333461716.jpg //czaszka
// https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlRbMEWZm-riYvJZxSSFr4Zakq5yETngeL7dG2nm-eMQF8tc7X //czerwona
// i przerobiona w GIMP


#include <cv.hpp>
#include <highgui.h>
#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
Mat hat;
Mat scarf;
vector <vector<Point2f>> detectAndDisplay( Mat frame );

/** Functions **/
vector <vector<Point2f>> detectAndDisplay( Mat frame ) {
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
//imshow ( "frame_gray", frame_gray );

	// detect face
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size( 12, 12 ) );

	for( size_t i = 0; i < faces.size(); i++ ) {
		Mat faceROI = frame_gray( faces[i] ); // range of interest
//imshow ( "ROI Face", faceROI );
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 1, 0 | CASCADE_SCALE_IMAGE, Size( 7, 7 ) );
		if( eyes.size() > 0 ) {
			vector<Point2f> dst1 = {
/*
				Point2f( faces[i].x, faces[i].y + faces[i].height * 5 / 20 ) ,
				Point2f( faces[i].x + faces[i].width, faces[i].y + faces[i].height * 5 / 20               ) ,
				Point2f( faces[i].x + faces[i].width, faces[i].y + faces[i].height * 5 / 20 + faces[i].height * 3 / 10 ) ,
				Point2f( faces[i].x, faces[i].y + faces[i].height * 5 / 20 + faces[i].height * 3 / 10 )
*/
				Point2f( faces[i].x - faces[i].width / 4					, faces[i].y - faces[i].height + faces[i].height / 3 ) ,
				Point2f( faces[i].x + faces[i].width + faces[i].width / 4	, faces[i].y - faces[i].height + faces[i].height / 3 ) ,
				Point2f( faces[i].x + faces[i].width + faces[i].width / 4	, faces[i].y + faces[i].height / 3 ) ,
				Point2f( faces[i].x - faces[i].width / 4					, faces[i].y + faces[i].height / 3 )
			};
			
			vector<Point2f> dst2 = {
				Point2f( faces[i].x 						, faces[i].y + faces[i].height * 3 / 8 ) ,
				Point2f( faces[i].x + faces[i].width * 1.2 	, faces[i].y + faces[i].height * 3 / 8 ) ,
				Point2f( faces[i].x + faces[i].width * 1.2	, faces[i].y + faces[i].height * 1.5 ) ,
				Point2f( faces[i].x							, faces[i].y + faces[i].height * 1.5 )
			};
			
			vector <vector<Point2f>> dst = { dst1, dst2 };
			return dst;
		}
	}
	return {{},{}};
}

// funkcja nakladajaca obraz z przezroczystoscia
// w oparciu o http://dsynflo.blogspot.in/2014/08/simplar-2-99-lines-of-code-for.html
void imageOverImageBGRA( const Mat &srcMat, Mat &dstMat, const vector<Point2f> &dstFrameCoordinates ) {
	if ( srcMat.channels() != 4 ) throw "Nakladam tylko obrazy BGRA";

	// tylko kanal alpha
	vector<Mat> rgbaChannels( 4 );
	Mat srcAlphaMask( srcMat.rows, srcMat.cols, srcMat.type() );
	split( srcMat, rgbaChannels );
	rgbaChannels = {rgbaChannels[3],rgbaChannels[3],rgbaChannels[3]};
	merge( rgbaChannels, srcAlphaMask );

	// wspolrzedne punktow z obrazu nakladanego
	vector<Point2f> srcFrameCoordinates = {{0,0},{(float)srcMat.cols,0},{(float)srcMat.cols,(float)srcMat.rows},{0,(float)srcMat.rows}};
	Mat warp_matrix = getPerspectiveTransform( srcFrameCoordinates, dstFrameCoordinates );

	Mat cpy_img( dstMat.rows, dstMat.cols, dstMat.type() );
	warpPerspective( srcAlphaMask, cpy_img, warp_matrix, Size( cpy_img.cols, cpy_img.rows ) );
	Mat neg_img( dstMat.rows, dstMat.cols, dstMat.type() );
	warpPerspective( srcMat, neg_img, warp_matrix, Size( neg_img.cols, neg_img.rows ) );
	dstMat = dstMat - cpy_img;
	
	cvtColor(neg_img, neg_img, CV_BGRA2BGR);
	cpy_img = cpy_img / 255;
	neg_img = neg_img.mul( cpy_img );
	dstMat = dstMat + neg_img;
}

// funkcja okreslajaca obszary
// w oparciu o http://opencvexamples.blogspot.com/2013/10/calculating-moments-of-image.html
Point2f find_moments( Mat canny_output, bool adjusting ) {
//    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
//	RNG rng(12345); random number generator for color

    /// Detect edges using canny
//    Canny( gray, canny_output, 50, 150, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  
	//object not found
	if ( contours.size() < 1 ) {
		vector<Point2f> mc(1);
		return mc[1];
	}
	
    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ ) {
		mu[i] = moments( contours[i], false );
	}
 
    ///  Get the mass centers:
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ ) {
		if (mu[i].m00 == 0) {
			return Point2f(0,0);
		}
		mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
	}
 
    /// search max area element
    int max_i = 0;
    for( int i = 1; i< contours.size(); i++ ) {
		if (contourArea(contours[i]) > contourArea(contours[max_i])) {
			max_i = i;
		}
	}
	if (adjusting) {
		printf("max area found at i = [%d]\n", max_i );
	}

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

	{ //only here we need "i" variable
	int i = max_i;
//    for( int i = 0; i< contours.size(); i++ ) {
//        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        Scalar color = Scalar( 0, 255, 0 );//green
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        circle( drawing, mc[i], 4, color, -1, 8, 0 );
//    }
	}
 
    /// Show in a window
    //namedWindow( "CONTOURS", CV_WINDOW_AUTOSIZE );
	if (adjusting) {
		imshow( "CONTOURS", drawing );
 
    /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
		for( int i = 0; i< contours.size(); i++ ) {
			printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
		}
	}
 
	return mc[max_i];
 
}


/**
 * @function main
 */
int main( void ) {
	VideoCapture capture;
	Mat frameSRC;
	Mat frame;
	Mat frameHSV;
	
	//wl/wyl podgladu i zestawu suwakow do kalibracji szukanego koloru
//	bool adjusting = true;
	bool adjusting = false;

	//-- 1. Load the cascade
	if( !face_cascade.load( String( "lbpcascade_frontalface.xml" ) ) ) {
//	if( !face_cascade.load( String( "lbpcascade_frontalcatface.xml" ) ) ) { //too big delay
		return -9;
	};
	if( !eyes_cascade.load( String( "haarcascade_eye_tree_eyeglasses.xml" ) ) ) { //face
//	if( !eyes_cascade.load( String( "haarcascade_frontalcatface.xml" ) ) ) { //mount%nose ?? face ??
		return -8;
	};
//	glasses = imread( "dwi.png", -1 ); // glasses
	hat = imread( "hat.png", -1 ); //hat
	scarf = imread( "scarf.png", -1 ); //scarf
	if (adjusting) {
		std::cout << "C hat:" << hat.channels() << "\n"; // check number of channels
		std::cout << "C scarf:" << scarf.channels() << "\n";
	}
	
	capture.open( -1 );
	if ( ! capture.isOpened() ) {
		return -7;
	}
	
	//RED
	int iLowH = 125; //0
	int iHighH = 179; //179
	int iLowS = 149; //0
	int iHighS = 255; //255
	int iLowV = 0; //0
	int iHighV = 255; //255
	if (adjusting) {
		namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
		cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
		cvCreateTrackbar("HighH", "Control", &iHighH, 179);
		cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
		cvCreateTrackbar("HighS", "Control", &iHighS, 255);
		cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
		cvCreateTrackbar("HighV", "Control", &iHighV, 255);
	}

	while ( capture.read( frameSRC ) ) {
		if( frameSRC.empty() ) return -1;
			cv::flip( frameSRC, frameSRC, 1 );
		frame = frameSRC.clone();

		if (adjusting) {
			imshow( "SRC", frame );//RAW source frame
		}

		//PART WITH TRACKING FACE
		auto detected_obj = detectAndDisplay( frame );
		if (detected_obj[0].size() && detected_obj[1].size()) {
			imageOverImageBGRA( hat.clone(), frame, detected_obj[0] );
			imageOverImageBGRA( scarf.clone(), frame, detected_obj[1] );
			imshow( "KOWBOY: FREEZE-FRAME", frame );
		}
		else { //none displayed if face not detected
//			imshow( "DWI", frame );
		}
		
		//PART WITH TRACKING OBJECT
		frame = frameSRC.clone(); // redefined frame;
		cvtColor(frame, frameHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded;
		inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
		  
		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

		//morphological closing (fill small holes in the foreground)
		dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

		if (adjusting) {
			imshow("Thresholded Image", imgThresholded); //show the thresholded image
		}
		
		Mat frame_gray;
		cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );

		Point2f znaleziony = find_moments( imgThresholded, adjusting );
		if (znaleziony.x > 0 && znaleziony.y > 0) {
			//rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
			int wielkosc_zaznaczenia = 40;
			rectangle(frame, Point(znaleziony.x - wielkosc_zaznaczenia, znaleziony.y - wielkosc_zaznaczenia), Point(znaleziony.x + wielkosc_zaznaczenia, 
			znaleziony.y + wielkosc_zaznaczenia), Scalar(0,255,0), 2, 4, 0 );
			imshow("TRACKED RED OBJECT", frame); //show border
		}
		
		if( (waitKey( 1 )&0x0ff) == 27 ) return 0;
	}
	return 0;
}
