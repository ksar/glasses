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
#include <list>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
Mat hat;
Mat scarf;

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
//TEMPORARY
//		printf("max area found at i = [%d]\n", max_i );
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
//TEMPORARY
//			printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
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
	Mat frameSYM;//frame for symbol detection
	
	//wl/wyl podgladu i zestawu suwakow do kalibracji szukanego koloru
	bool adjusting = true;
//	bool adjusting = false;

//figura N-Z
	int nr_linii = 0;
	int licznik = 0;
	int limit = 10;
/*
	vector<pair<int, int>> linia(limit);
	linia[1].first  = 2;
	linia[1].second = 3;
	linia[2].first  = 4;
	linia[2].second = 5;
*/
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
	int iLowS = 200; //0 149
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
	
	//SYMBOL DETECTION
	const int imgwidth = 512, imgheight = 384;
	const int dilation_size = 2;
	list < Point2f > path;


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

		frameSYM = imgThresholded.clone(); //copy for symbol detection - after conversion to HSV and color filter

		if (adjusting) {
			imshow("THRESHOLDED", imgThresholded); //show the thresholded image
		}
		
		Mat frame_gray;
		cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );

		Point2f znaleziony = find_moments( imgThresholded, adjusting );
		
		if (adjusting) {
//			std::cout << "x:" << znaleziony.x << "  y:" << znaleziony.y << "\n";
		}

/*
		linia[licznik].first  = (int)znaleziony.x;
		linia[licznik].second = (int)znaleziony.y;
		vector < pair < double, vector < Point > > > sortedContours;
		sortedContours = znaleziony;
		vector < vector < Point > > contours;
		
		if (linia[licznik].first > 0 && linia[licznik].second > 0) {
			//horizontal line
			int maxx = 0, minx = 9999;
			int maxy = 0, miny = 9999;
			for (int t = 0 ; t <= limit ; t++) {
				maxx = ( linia[t].first  > maxx ? linia[t].first  : maxx );
				minx = ( linia[t].first  < minx ? linia[t].first  : minx );
				maxy = ( linia[t].second > maxy ? linia[t].second : maxy );
				miny = ( linia[t].second < miny ? linia[t].second : miny );
//				std::cout << "t:" << t << "\n";
//				std::cout << " # minx:" << minx << " # maxx:" << maxx << " # miny:" << miny << " # maxy:" << maxy << "\n";
			}

			if ((maxy - miny) < 50 && (maxx - minx) > 300) { //dispersion of y and x
				nr_linii = 1; // first line detected
			}
			
//			std::cout << "dispX:" << (maxx-minx) << " # dispY:" << (maxy-miny) << "\n";
			
			licznik++;
			if (licznik>=limit) licznik=0;
		}
		if (adjusting) {
//			std::cout << "nr_linii:" << nr_linii << "  # licznik:" << licznik << " # linia x:" << linia[licznik].first << " # linia y:" << linia[licznik].second << "\n";
		}
*/

		//draw rectangle on detected object
		if (znaleziony.x > 0 && znaleziony.y > 0) {
			//rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
			int wielkosc_zaznaczenia = 40;
			rectangle(frame, Point(znaleziony.x - wielkosc_zaznaczenia, znaleziony.y - wielkosc_zaznaczenia), Point(znaleziony.x + wielkosc_zaznaczenia, 
			znaleziony.y + wielkosc_zaznaczenia), Scalar(0,255,0), 2, 4, 0 );
			imshow("TRACKED RED OBJECT", frame); //show border
		}

//SYMBOL DETECTION (Z+N)
// below code based on:
//https://github.com/pantadeusz/examples-ai/openvc/example2/cv2.cpp
		resize(frameSYM, frameSYM,{imgwidth, imgheight});
//		cvtColor(frameSYM, frameSYM, CV_RGB2HSV);
		vector < vector < Point > > contours;

		findContours(frameSYM, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		vector < pair < double, vector < Point > > > sortedContours;
		for (unsigned i = 0; i < contours.size(); i++) {
			drawContours(frameSYM, contours, i, Scalar(255,0,0),3);
			sortedContours.push_back({contourArea(contours[i], false),contours[i] });
		}
		if (sortedContours.size() > 0) {
			Point2f pos;
			float r;
			sort(sortedContours.begin(),sortedContours.end(),[](auto a, auto b){
				return a.first > b.first;
			});
			minEnclosingCircle(sortedContours[0].second, pos, r);
			if (r > 8) {
				if (path.size() < 70) {
					path.push_back(pos); // dopisujemy srodek okregu
				} else {
					path.pop_front();
					path.push_back(pos);
				}
				vector < Point > pathV;
				vector < Point2f > approximated;
				approxPolyDP(vector<Point2f>(path.begin(), path.end()),approximated,50, false);
				
				for (auto &p: approximated) pathV.push_back({(int)p.x,(int)p.y});
				polylines(frame,{pathV},false,Scalar(0,0,255),2); // jesli chcemy pokazać ścieżkę

				//Z
				if (pathV.size() >= 4) {
					vector < Point > itr(pathV.end()-4,pathV.end());
					int conditions = 0;
					double factor = (::abs(itr[0].x - itr[1].x) + ::abs(itr[0].y - itr[1].y))*2/3;
					if ((::abs(itr[0].x - itr[1].x) > factor) && (::abs(itr[0].y - itr[1].y) < factor)) {
						conditions++;
					}
					if ((::abs(itr[1].x - itr[2].x) > factor) && (::abs(itr[1].y - itr[2].y) > factor)) {
						conditions++;
					}
					if ((::abs(itr[2].x - itr[3].x) > factor) && (::abs(itr[2].y - itr[3].y) < factor)) {
						conditions++;
					}
					if (conditions == 3) {
						cout << "Jest Z!!!" << endl;
						path.clear();
					}
					cout << "Z: " << conditions << "  factor = " << factor << endl;
				}

				//N
				if (pathV.size() >= 4) {
					vector < Point > itr2(pathV.end()-4,pathV.end());
					int conditions = 0;
					double factor = (::abs(itr2[0].x - itr2[1].x) + ::abs(itr2[0].y - itr2[1].y))*2/3;
					if ((::abs(itr2[0].x - itr2[1].x) < factor) && (::abs(itr2[0].y - itr2[1].y) > factor)) {
						conditions++;
					}
					if ((::abs(itr2[1].x - itr2[2].x) < factor) && (::abs(itr2[1].y - itr2[2].y) < factor)) {
						conditions++;
					}
					if ((::abs(itr2[2].x - itr2[3].x) < factor) && (::abs(itr2[2].y - itr2[3].y) > factor)) {
						conditions++;
					}
					if (conditions == 3) {
						cout << "Jest N!!!" << endl;
						path.clear();
					}
					cout << "N: " << conditions << "  factor = " << factor << endl;
				}
			}
		}
//		imshow("GEST DETECTOR", frameSYM);
		imshow("TRACKED RED OBJECT", frame); //show show path

		if( (waitKey( 1 )&0x0ff) == 27 ) return 0;
	}
	return 0;
}
