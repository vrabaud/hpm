#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "file.hpp"
#include "draw.hpp"

#include "hpm/util.hpp"
#include "hpm/hpmatch.hpp"

using namespace std;
using namespace cv;
using namespace hpm;

//------------------------------------------------------------------------
// clear matches whose NN ratio is too large

void ratio_test(vector<vector<DMatch> >& matches, float ratio = 0.65f)
{
	for(size_t i=0; i<matches.size(); i++) {
		vector<DMatch>& m = matches[i];
		if((m.size() > 1) && (m[0].distance / m[1].distance > ratio))
			m.clear();
	}
}

//------------------------------------------------------------------------
// HPM match colors, according to strengths and erase / out-of-bound
// (invalid) status

vector<Scalar> match_colors(const vector<double>& strengths,
	const vector<bool>& erased, const vector<bool>& out, double lambda = 1e-4,
	Scalar strong = Scalar(0,0,255), Scalar weak = Scalar(0,255,255),
	Scalar invalid = Scalar(255,255,0))
{
	vector<Scalar> colors(strengths.size());
	vector<double> val;
	for(size_t i=0;i<strengths.size();i++)
		val.push_back( exp(-lambda * strengths[i] * strengths[i]) );

	for(size_t i=0; i<colors.size(); i++)
		colors[i] = erased[i] || out[i] ? invalid :
			val[i] * weak + (1-val[i]) * strong;

	return colors;
}

//------------------------------------------------------------------------
// image matching with HPM

void match(bool extract)
{
	string img1f, img2f;                // input image files
	Mat img1, img2;                     // input images

	vector<KeyPoint> points1, points2;  // keypoints (cv style)
	vector<vector<DMatch> > matches;    // feature matches (cv style)

	vector<size_t> labels1, labels2;        // feature labels

	Mat disp;                           // image for display

//------------------------------------------------------------------------
// load images

	img1f = "ec1m_00130055.jpg";
	img2f = "ec1m_00130056.jpg";

	img1 = imread(img1f);
	img2 = imread(img2f);

//------------------------------------------------------------------------
// extract features, descriptors + correspondences from images ...

	if(extract)
	{
		SURF surf(2500.0);
		Mat desc1, desc2;

		surf(img1, Mat(), points1, desc1);
		surf(img2, Mat(), points2, desc2);

		BFMatcher matcher(NORM_L2);
		matcher.knnMatch(desc1, desc2, matches, 2);
		ratio_test(matches);

		drawMatches(img1, points1, img2, points2, matches, disp,
			Scalar(0,255,255), Scalar::all(-1), vector<vector<char> >(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("tentative matches", disp);
		waitKey();
	}

//------------------------------------------------------------------------
// ... or, load features + labels from files

	else
	{
		string feat1f = "ec1m_00130055.jpg.surf.bin";
		string feat2f = "ec1m_00130056.jpg.surf.bin";

		vector<vector<float> > feat1a;
		vector<vector<float> > feat2a;
		load_bin(feat1a,feat1f);
		load_bin(feat2a,feat2f);

		vector<size_t> idxf;
		idxf.push_back(0); idxf.push_back(1); idxf.push_back(3); idxf.push_back(4);
		feat1a = get_columns(feat1a, idxf);
		feat2a = get_columns(feat2a, idxf);

		vector<feat> feat1 = feat::vector2feat(feat1a);
		vector<feat> feat2 = feat::vector2feat(feat2a);

		points1 = feat::feat2key(feat1);
		points2 = feat::feat2key(feat2);

		for(size_t i=0; i<points1.size(); i++) points1[i].size *= 5;
		for(size_t i=0; i<points2.size(); i++) points2[i].size *= 5;

		string lab1f = "ec1m_00130055.jpg.surf.surf.bin.lbl.bin";
		string lab2f = "ec1m_00130056.jpg.surf.surf.bin.lbl.bin";

		vector<unsigned int> labels1a, labels2a;
		load_bin(labels1a, lab1f);
		load_bin(labels2a, lab2f);

		vector<size_t> labels1t(labels1a.begin(), labels1a.end());
		vector<size_t> labels2t(labels2a.begin(), labels2a.end());

		labels1 = labels1t;
		labels2 = labels2t;
	}

//------------------------------------------------------------------------
// HPM matching

	HPMatcher H;              // HPM matcher
	double sim;               // similarity score between two images

	vector<double> strengths; // individual feature strengths
	vector<bool> erased;	   // erased correspondences
	vector<bool> out;		   // correspondences out of bound

	if(extract)  // matches are given
		sim = H.match(points1, points2, matches, strengths, erased, out);

	else  // matches are found from labels
		sim = H.match(points1, labels1, points2, labels2, matches, strengths, erased, out);

	cout << "similarity: " << sim << endl;

//------------------------------------------------------------------------
// visualize results

	vector<DMatch> ord_matches = order_by(implode(matches), strengths);
	vector<Scalar> colors = order_by(match_colors(strengths, erased, out), strengths);

	drawMatches(img1, points1, img2, points2, ord_matches, disp,
		colors, Scalar(0,255,0), Scalar::all(-1), vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS, 2);

	imshow("HPM matches", disp);
	waitKey();
}

//------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	// extract features if no argument given; load from files otherwise
	match(argc == 1);

	return 0;
}

