#ifndef _HPMATCH_HPP_
#define _HPMATCH_HPP_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "feature.hpp"
#include "corr.hpp"

namespace hpm {
using namespace cv;

//------------------------------------------------------------------------------
// hough pyramid matcher

class HPMatcher
{
private:

	size_t L, B;               // number of levels, number of bins
	vector<double> idf;         // inverse document frequency (idf) array
	range<double> rt, rs, rr;  // translation, scale, rotation ranges
	unsigned int z[3];         // bases for encoding 4d<->1d; b3 most significant

public:

	HPMatcher(size_t l = 5,    // number of levels
		double mt = 500*3.5,    // max translation (typically, 3.5 times the max image size)
		double ms = 10.0,       // max relative scale between the two images
		const vector<double>& i
			= vector<double>())   // idf vector (at vocabulary size)

		:

		L(l), B(1 << (l-1)), idf(i),
		rt(-mt,      2.0 * mt / B,        mt),
		rs(1.0 / ms, 2.0 * log10(ms) / B, ms),
		rr(0.0,      2.0 * pi / B,        2.0 * pi)

		{ z[0] = B; z[1] = z[0] * B; z[2] = z[1] * B; }

	// match given feature labels (visual words) (cv style)
	double match(const vector<KeyPoint>& points1, const vector<size_t>& lab1,
		const vector<KeyPoint>& points2, const vector<size_t>& lab2,
		vector<vector<DMatch> >& matches, vector<double>& strengths,
		vector<bool>& erased, vector<bool>& out_of_bounds);

	// match given feature correspondences (cv style)
	double match(const vector<KeyPoint>& points1, const vector<KeyPoint>& points2,
		const vector<vector<DMatch> >& matches, vector<double>& strengths,
		vector<bool>& erased, vector<bool>& out_of_bounds);
private:

	// match given feature labels (visual words)
	double match(const vector<feat>& feat1, const vector<size_t>& lab1,
		const vector<feat>& feat2, const vector<size_t>& lab2,
		vector<vector<size_t> >& cor, vector<double>& strengths,
		vector<bool>& erased, vector<bool>& out_of_bounds);

	// match given feature correspondences
	double match(const vector<feat>& feat1, const vector<feat>& feat2,
		const vector<vector<size_t> >& cor, vector<double>& strengths,
		vector<bool>& erased, vector<bool>& out_of_bounds);

	// HPM, given both labels and correspondences
	double hpm(const vector<feat>& feat1, const vector<size_t>& lab1,
		const vector<feat>& feat2, const vector<size_t>& lab2,
		const vector<vector<size_t> >& cor, vector<double>& strengths,
		vector<bool>& erased, vector<bool>& out_of_bounds);

	// recursive HPM
	void hpm_rec(const size_t l, const vector<unsigned int>& trans, const vector<size_t>& lab,
		const vector<size_t>& index, vector<vector<size_t> >& g, vector<float>& s,
		vector<bool>& X, vector<std::vector<std::vector<size_t> > >& sib);
};

} // namespace hpm

#endif  // _HPMATCH_HPP_