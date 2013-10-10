#include "hpmatch.hpp"

using namespace std;
using namespace cv;

namespace hpm {

template<>
const unsigned int feat::oob = -1;

//---------------------------------------------------------------------------
// main HPM entry point when feature labels (visual words) are known
// (cv style)

double HPMatcher::match(const vector<KeyPoint>& points1,
						const vector<size_t>& lab1,
						const vector<KeyPoint>& points2,         // features [x y scale rotation]
						const vector<size_t>& lab2,                  // labels (visual words)
						vector<vector<DMatch> >& matches,        // tentative correspondences
						vector<double>& strengths,                // correspondence strengths
						vector<bool>& erased,                     // erased correspondences
						vector<bool>& out_of_bounds)              // out of bound transformations
{
	vector<vector<size_t> > cor;
	double sim = match(feat::key2featv(points1), lab1,
							 feat::key2featv(points2), lab2,
							 cor, strengths, erased, out_of_bounds);
	matches = corr::cor2match(cor);
	return sim;
}

//---------------------------------------------------------------------------
// main HPM entry point when correspondences are known (cv style)

double HPMatcher::match(const vector<KeyPoint>& points1,
						const vector<KeyPoint>& points2,         // features [x y scale rotation]
						const vector<vector<DMatch> >& matches,  // tentative correspondences
						vector<double>& strengths,                // correspondence strengths
						vector<bool>& erased,                     // erased correspondences
						vector<bool>& out_of_bounds)              // out of bound transformations
{
	return match(feat::key2featv(points1), feat::key2featv(points2),
					 corr::match2corv(matches), strengths, erased, out_of_bounds);
}

//---------------------------------------------------------------------------
// main HPM entry point when feature labels (visual words) are known,
// but not correspondences. just find correspondences and call inner HPM.

double HPMatcher::match(const vector<feat>& feat1,
						const vector<size_t>& lab1,
						const vector<feat>& feat2,                // features [x y scale rotation]
						const vector<size_t>& lab2,                  // labels (visual words)
						vector<vector<size_t> >& cor,                  // tentative correspondences
						vector<double>& strengths,                // correspondence strengths
						vector<bool>& erased,                     // erased correspondences
						vector<bool>& out_of_bounds)              // out of bound transformations
{
	// find correspondences from labels
	cor = corr::lab2cor(lab1, lab2);

	// call inner HPM, giving both labels and correspondences
	return hpm(feat1, lab1, feat2, lab2, cor, strengths, erased, out_of_bounds);

	// test/debug: conversely, find labels again from correspondences
	//return hpm(feat1, feat2, cor, strengths, erased, out_of_bounds);
}

//---------------------------------------------------------------------------
// main HPM entry point when correspondences are known, but not
// feature labels (visual words). just find labels and call inner HPM.

double HPMatcher::match(const vector<feat>& feat1,
						const vector<feat>& feat2,                // features [x y scale rotation]
						const vector<vector<size_t> >& cor,            // tentative correspondences
						vector<double>& strengths,                // correspondence strengths
						vector<bool>& erased,                     // erased correspondences
						vector<bool>& out_of_bounds)              // out of bound transformations
{
	// find components (labels / visual words) from correspondences
	vector<size_t> lab1, lab2;
	corr::cor2lab(cor, lab1, lab2);

	// call inner HPM, giving both labels and correspondences
	return hpm(feat1, lab1, feat2, lab2, cor, strengths, erased, out_of_bounds);
}

//---------------------------------------------------------------------------
// inner HPM call when both labels and correspondences are known.
// given two input vectors of features and corresponding vectors of labels
// (visual words) and correspondences, quantize relative transformations,
// call recursive HPM, and return correspondences, strengths, and vectors
// of correspondences that are erased or out of (transformation space)
// bounds.

double HPMatcher::hpm(const vector<feat>& feat1,
						const vector<size_t>& lab1,
						const vector<feat>& feat2,      // features [x y scale rotation]
						const vector<size_t>& lab2,        // labels (visual words)
						const vector<vector<size_t> >& cor,  // tentative correspondences
						vector<double>& strengths,      // correspondence strengths
						vector<bool>& erased,           // erased correspondences
						vector<bool>& out_of_bounds)    // out of bound transformations
{
	// align features based on tentative correspondences
	vector<feat> af1, af2;
	corr::align(feat1, feat2, cor, af1, af2);

	// at least two correspondences needed to match the images
	if(!af1.size()) return 0.0;

	// align correspondence labels (visual words)
	vector<size_t> lab;
	corr::align(lab1, lab2, cor, lab, lab);

	// relative transformations
	vector<feat> tran = feat::rel_tran(af1, af2);

	// quantized transformations within bounds
	vector<size_t> kept;
	vector<unsigned int> qtran = feat::quant(tran, kept, rt, rs, rr, z);

	// number of correspondence transformations kept
	size_t C = qtran.size();

	vector<vector<size_t> > g(C);                  // group count of each correspondence at each level
	vector<float> s(C, 0.0);                  // correspondence strengths
	vector<bool> X(C, false);                 // erased correspondences
	vector<vector<vector<size_t> > > sib(L);  // sibling groups at each level

	// initialize groups
	for(size_t i=0; i<g.size(); i++) g[i].resize(L, 0);

	// recursive HPM call
	vector<size_t> rng;
	for(size_t i=0;i<C;i++)
		rng.push_back(i);
	if(C) hpm_rec(0, qtran, lab, rng, g, s, X, sib);

	// update group count of correspondences in same bin with erased ones
	// "undoing" contribution of correspondences that are eventually erased
	for(size_t l=0; l<L; l++)
		for(size_t i=0, x; x = 0, i<sib[l].size(); i++)
		{
			const vector<size_t>& si = sib[l][i];
			for(size_t j=0; j<si.size(); j++) if(X[si[j]]) x++;
			for(size_t j=0; j<si.size(); j++) if(!X[si[j]]) g[si[j]][l] -= x;
		}

	// correspondence strengths; erased ones have zero strength
	strengths.resize(af1.size(), 0.0);
	for(size_t i=0; i<C; i++) if(!X[i])
		for(size_t l=0; l<L && g[i][l] > 0; l++)
			strengths[kept[i]] += g[i][l] << (l ? (l-1) : 0);

	// weigh strengths by idf
	if(idf.size()) 
		for(size_t i=0;i<kept.size();i++)
			strengths[kept[i]] *= idf[lab[kept[i]]];

	// positions of erased correspondences
	erased.resize(tran.size(), false);

	for(size_t i=0;i<X.size();i++)
		if(X[i])
			erased[kept[i]] = true;
	
	// positions of correspondences out of bounds
	out_of_bounds.resize(tran.size(), true);
	for(size_t i=0;i<kept.size();i++)
		out_of_bounds[kept[i]] = false;
	
	double sm = 0.0;
	for(size_t i=0;i<strengths.size();i++)	sm+= strengths[i];
	// similarity score
	return sm;
}

//-----------------------------------------------------------------------------
// recursive part of HPM. given current level, encoded correspondence
// transformations, their visual words and positions in initial vector,
// call HPM recursively to quantize and group correspondences at all levels
// below, then update their group count, strengths, and vector of erased
// correspondences. also update "sibling" groups to be used in "undoing"
// strength contributions from correspondences that are eventually erased.

// level l starts from 0 and increases through recursion down the pyramid.
// l = 0 corresponds to the top, coarsest level, with only one bin.
// l = L-1 corresponds to the bottom, finest level.

void HPMatcher::hpm_rec(const size_t l,                     // current level in pyramid
					const vector<unsigned int>& qtran,      // encoded correspondence transformations
					const vector<size_t>& lab,                 // label (visual word) of each correspondence
					const vector<size_t>& index,               // current subset of correspondences
					vector<vector<size_t> >& g,                  // group count of each correspondence at each level
					vector<float>& s,                       // correspondence strengths
					vector<bool>& X,                        // erased correspondences
					vector<vector<vector<size_t> > >& sib)  // sibling groups at each level
{
	// groups of correspondences in each bin, and their count
	vector<vector<size_t> > bins;
	vector<size_t> bc;

	// quantize transformations at current level and group them in bins
	unique_count(feat::quant(qtran, z, L-1-l), bc, bins);

	// if not at bottom, recursively call hpm on bins with more than one correspondence
	if(l < L-1) for(size_t i=0; i<bins.size(); i++) if(bc[i] > 1)
	{
		const vector<size_t>& b = bins[i];
		vector<unsigned int> qtrant;
		vector<size_t> labt;
		vector<size_t> indext;
		for(size_t i=0;i<b.size();i++)
		{
			qtrant.push_back(qtran[b[i]]);
			labt.push_back(lab[b[i]]);
			indext.push_back(index[b[i]]);
		}

		hpm_rec(l+1, qtrant, labt, indext, g, s, X, sib);
	}

	// loop over bins with more than one correspondence
	for(size_t i=0; i<bins.size(); i++) if(bc[i] > 1)
	{
		const vector<size_t>&b = bins[i];  // group of current bin
		
		vector<size_t> labt;
		for(size_t i=0;i<b.size();i++)
			labt.push_back(lab[b[i]]);
			
		// conflict groups in current bin, and their count
		vector<vector<size_t> > conflicts;
		vector<size_t> cc;

		// unique visual words in current bin
		//vector<size_t> ulab = unique_count(lab[b], cc, conflicts);
		vector<size_t> ulab = unique_count(labt, cc, conflicts);

		// there is exactly one representative correspondence for each
		// unique visual word
		if(ulab.size() < 2) continue;

		// update correspondence group counts and strengths, unless erased
		for(size_t j=0, cj; j<b.size(); j++)
			if(!X[cj = index[b[j]]])
				s[cj] += (g[cj][l] = ulab.size()-1) << (l ? (l-1) : 0);

		// siblings in current bin
		sib[l].push_back(vector<size_t>());
		vector<size_t>& si = sib[l].back();

		// loop over unique visual words in current bin
		for(size_t u=0; u<cc.size(); u++)
		{
			// keep position of strongest correspondence in conflict group
			//vector<float> wgs = s[index[ b[conflicts[u]] ]];
			vector<float> wgs;
			for(size_t k=0;k<conflicts[u].size();k++)
				wgs.push_back(s[index[b[conflicts[u][k]]]]);

			//size_t strong = find(wgs == max(wgs))[0];
			size_t strong = 0;
			for(size_t k=0;k<wgs.size();k++)
				if(wgs[strong]<wgs[k])
					strong = k;
			
			si.push_back(index[ b[conflicts[u][strong]] ]);
 
			// erase all other correspondences
			for(size_t k=0; k<cc[u]; k++)
				if(k != strong) X[index[ b[conflicts[u][k]] ]] = true;
		}
	}
}

} // namespace hpm
