#include <hpm/private/corr.hpp>

namespace hpm {
namespace corr {

//------------------------------------------------------------------------------
// descriptor matches -> correspondences

vector<vector<size_t> > match2corv(const vector<vector<cv::DMatch> >& matches)
{
	vector<vector<size_t> > cor(matches.size());
	for(size_t i=0; i<matches.size(); i++) {
		cor[i].resize(matches[i].size());
		for(size_t j=0; j<matches[i].size(); j++)
			cor[i][j] = matches[i][j].trainIdx;
	}
	return cor;
}

} // namespace corr
} // namespace hpm
