#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace cv
{

using namespace std;

// Draws matches of keypints from two images on output image.
void drawMatches( const Mat& img1, const vector<KeyPoint>& keypoints1,
						const Mat& img2, const vector<KeyPoint>& keypoints2,
						const vector<DMatch>& matches1to2, Mat& outImg,
						const vector<Scalar>& matchColors, 
						const Scalar& pointColor=Scalar::all(-1),
						const Scalar& singlePointColor=Scalar::all(-1),
						const vector<char>& matchesMask=vector<char>(), int flags=DrawMatchesFlags::DEFAULT, float width=1 );

}  // namespace cv
