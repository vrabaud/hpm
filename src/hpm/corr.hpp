#ifndef _COR_HPP_
#define _COR_HPP_

#include <vector>
#include <opencv2/features2d/features2d.hpp>

#include "util.hpp"

namespace hpm {
namespace corr {

using namespace std;


//------------------------------------------------------------------------------
// find correspondences from labels (visual words)

template<typename T>
vector<vector<size_t> > lab2cor(const vector<T>& lab1, const vector<T>& lab2)
{
	size_t sz = lab1.size();
	vector<T> common;
	vector<vector<size_t> > ia, ib;
	intersect(lab1, lab2, common, ia, ib);

	// initialize array with correspondences
	vector<vector<size_t> > cor(sz);
	for(size_t i=0; i<sz; i++) cor[i].clear();

	// add correspondences
	for(size_t i=0; i<common.size(); i++)
		for(size_t j=0; j<ia[i].size(); j++)
			for(size_t k=0; k<ib[i].size(); k++)
				cor[ia[i][j]].push_back(ib[i][k]);
	return cor;
}

// connected components (disjoint set forest + union-find)
template<typename T>
struct component {
	component *p;  // parent
	size_t r;      // rank
	T l;           // load

	component() : p(0), r(0) { }
	inline component* set() { return p ? p = p->set() : this; }
	inline void join(component* c)
		{ if(r < c->r) p = c; else { c->p = this; if(r == c->r) r++; } }
};

typedef component<size_t> comp;

//------------------------------------------------------------------------------
// find components (labels / visual words) from correspondences

template<typename T>
void cor2lab(const vector<vector<size_t> >& cor, vector<T>& lab1, vector<T>& lab2)
{
	// count items (features) found in correspondences
	size_t sz1 = cor.size(), sz2 = 0;
	for(size_t i=0; i<sz1; i++)
		for(size_t j=0; j<cor[i].size(); j++)
			if(cor[i][j] > sz2) sz2 = cor[i][j];

	// assign a component to each item (feature)
	vector<comp> cc1(sz1), cc2(++sz2);
	for(size_t i=0; i<sz1; i++) cc1[i].l = i;
	for(size_t i=0; i<sz2; i++) cc2[i].l = sz1 + i;

	// connect components according to correspondences
	for(size_t i=0; i<sz1; i++)
		for(size_t j=0; j<cor[i].size(); j++) {
			comp *c1 = &cc1[i], *c2 = &cc2[cor[i][j]];
			if(c1->set() != c2->set()) c1->join(c2);
		}

	// return component labels
	lab1.resize(sz1);
	lab2.resize(sz2);
	for(size_t i=0; i<sz1; i++) lab1[i] = cc1[i].set()->l;
	for(size_t i=0; i<sz2; i++) lab2[i] = cc2[i].set()->l;
}

//------------------------------------------------------------------------------
// descriptor matches -> correspondences

vector<vector<size_t> > match2corv(const vector<vector<cv::DMatch> >& matches);

//------------------------------------------------------------------------------
// correspondences -> descriptor matches

template<typename T>
vector<vector<cv::DMatch> > cor2match(const vector<vector<T> >& cor)
{
	vector<vector<cv::DMatch> > match(cor.size());
	for(size_t i=0; i<cor.size(); i++)
		for(size_t j=0; j<cor[i].size(); j++)
			match[i].push_back(cv::DMatch(i, cor[i][j], 0));
	return match;
}

//------------------------------------------------------------
// align correspondences
// input:  1xK vector (K: number of vectors)
// output: 1xM vector (M: number of correspondences)

template<typename T, typename C> inline
void align(const vector<T> &x1, const vector<T> &x2,
			  const vector<vector<size_t> > &cor,
			  vector<C> &y1, vector<C> &y2)
{
	// count correspondences
	int ncor = 0;
	for(size_t i=0; i<cor.size(); i++) ncor += cor[i].size();

	// resize output vector
	y1.resize(ncor);
	y2.resize(ncor);

	// copy elements in aligned order
	for(size_t i=0, c=0; i<cor.size(); i++)
		for(size_t j=0; j<cor[i].size(); j++, c++)
		{
			y1[c] = x1[i];
			y2[c] = x2[cor[i][j]];
		}
};

} // namespace corr
} // namespace hpm

#endif  // _COR_HPP_
