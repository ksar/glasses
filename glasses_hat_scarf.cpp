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
/** Function Headers */
vector <vector<Point2f>> detectAndDisplay( Mat frame );
void imageOverImageBGRA( const Mat &srcMat, Mat &dstMat, const vector<Point2f> &dstFrameCoordinates );

/**
 * @function main
 */
int main( void ) {
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascade
	if( !face_cascade.load( String( "lbpcascade_frontalface.xml" ) ) ) {
//	if( !face_cascade.load( String( "lbpcascade_frontalcatface.xml" ) ) ) { //wieksze opoznienie
		return -9;
	};
	if( !eyes_cascade.load( String( "haarcascade_eye_tree_eyeglasses.xml" ) ) ) { //twarz
//	if( !eyes_cascade.load( String( "haarcascade_frontalcatface.xml" ) ) ) { //usta i nos ?? twarz ??
		return -8;
	};
//	glasses = imread( "dwi.png", -1 ); // okularki
	hat = imread( "hat.png", -1 ); //kapelusz
	scarf = imread( "scarf.png", -1 ); //chusta
	std::cout << "C hat:" << hat.channels() << "\n";
	std::cout << "C scarf:" << scarf.channels() << "\n";
	capture.open( -1 );
	if ( ! capture.isOpened() ) {
		return -7;
	}

	while ( capture.read( frame ) ) {
		if( frame.empty() ) return -1;
		
		auto detected_obj = detectAndDisplay( frame );
		
		if (detected_obj[0].size() && detected_obj[1].size()) {
			imageOverImageBGRA( hat.clone(), frame, detected_obj[0] );
			imageOverImageBGRA( scarf.clone(), frame, detected_obj[1] );
			imshow( "DWI", frame );
		}
		else {
//			cv::flip( frame, frame, 1 );
//			imshow( "DWI", frame );
		}
		if( (waitKey( 1 )&0x0ff) == 27 ) return 0;
	}
	return 0;
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
				Point2f( faces[i].x 						, faces[i].y + faces[i].height * 3 / 6 ) ,
				Point2f( faces[i].x + faces[i].width * 1.2 	, faces[i].y + faces[i].height * 3 / 6 ) ,
				Point2f( faces[i].x + faces[i].width * 1.2	, faces[i].y + faces[i].height * 2 ) ,
				Point2f( faces[i].x							, faces[i].y + faces[i].height * 2 )
			};
			
			vector <vector<Point2f>> dst = { dst1, dst2 };
			return dst;
		}
	}
	return {{},{}};
}
