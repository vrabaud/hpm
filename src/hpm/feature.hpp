#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <cmath>
#include <vector>
#include <opencv2/features2d/features2d.hpp>

#include "util.hpp"

static const double pi = 3.14159265358979323846;

namespace hpm {
using namespace std;

//------------------------------------------------------------------------------
// range of any type, with minimum, maximum values and a step

template<typename T>
struct range {
	T min, step, max;
	range() { }
	range(T min, T step, T max) : min(min), step(step), max(max) { }
};

//------------------------------------------------------------------------------
// feature, or similarity transformation

template<typename T>
struct feature {

	// x,y position (translation), (relative) scale, orientation (rotation)
	T x,y,s,r;

	// out-of-bounds code
	static const unsigned int oob;

	feature() {}

	// vector -> single feature
	template<typename S>
	feature(const vector<S>& a) : x(a[0]), y(a[1]), s(a[2]), r(a[3]) { }

	// keypoint -> single feature
	feature(const cv::KeyPoint& k) : x(k.pt.x), y(k.pt.y), s(k.size), r(k.angle*pi/180) { }

	// single feature -> keypoint
	operator cv::KeyPoint() const { return cv::KeyPoint(cv::Point2f(x,y), s, 180*r/pi); }

	// vector of vectors -> array of features
	template<typename S>
	static vector<feature> vector2feat(const vector<vector<S> >& a)
	{
		vector<feature> f(a.size());
		for(size_t i=0; i<a.size(); i++) f[i] = feature(a[i]);
		return f;
	}

	// vector of keypoints -> vector of features
	static vector<feature> key2featv(const vector<cv::KeyPoint>& k)
	{
		vector<feature> f(k.size());
		for(size_t i=0; i<k.size(); i++) f[i] = feature(k[i]);
		return f;
	}

	// vector of features -> vector of keypoints
	static vector<cv::KeyPoint> feat2key(const vector<feature>& f)
	{
		vector<cv::KeyPoint> v(f.size());
		for(size_t i=0; i<f.size(); i++) v[i] = f[i];
		return v;
	}

	// relative transformation from single correspondence
	feature(const feature& f1, const feature& f2)
	{
		r = fmod(f2.r - f1.r, 2.0 * pi);                   // rotation, in [0,2pi]
		s = f2.s / f1.s;                                  // relative scale
		T cos_r = cos(r), sin_r = sin(r);
		x = -s * f1.x * cos_r - s * f1.y * sin_r + f2.x;  // x translation
		y =  s * f1.x * sin_r - s * f1.y * cos_r + f2.y;  // y translation
		r = fmod(r + (5.0 * pi / 16.0), 2.0 * pi);         // shift rotations
	}

	// relative transformations from vector of correspondences
	static vector<feature> rel_tran(const vector<feature>& a1, const vector<feature>& a2)
	{
		vector<feature> tran(a1.size());
		for(size_t i=0; i<a1.size(); i++) tran[i] = feature(a1[i], a2[i]);
		return tran;
	}

	// quantize + encode single transformation
	unsigned int quant(const range<T>& rt, const range<T>& rs,
		const range<T>& rr, const unsigned int z[3]) const
	{
		// check bounds of relative location and scale
		if(x < rt.min || x >= rt.max || y < rt.min || y >= rt.max ||
			s < rs.min || s >= rs.max) return oob;

		// quantize parameters
		unsigned int q[4] = {
			(x - rt.min) / rt.step,              // x translation
			(y - rt.min) / rt.step,              // y translation
			floor(log10(s / rs.min) / rs.step),  // relative scale
			fmod(r, 2.0 * pi) / rr.step           // rotation
		};

		vector<unsigned int> qv (q, q + sizeof(q) / sizeof(unsigned int) );
		return encode(qv, z);
	}

	// quantize + encode array of transformations; keep those within bounds
	static vector<unsigned int> quant(const vector<feature>& tran, vector<size_t>& kept,
		const range<T>& rt, const range<T>& rs, const range<T>& rr,
		const unsigned int z[3])
	{
		vector<unsigned int> tv;       // quantized transformation vector
		vector<bool> inb(tran.size());  // subset of transformations within bounds

		// quantize transformations; keep them only if within bounds
		for(size_t i=0; i<tran.size(); i++)
		{
			unsigned int q = tran[i].quant(rt, rs, rr, z);
			if((inb[i] = (q != oob))) tv.push_back(q);
		}

		for(size_t i=0;i<inb.size();i++)
			if(inb[i])
				kept.push_back(i);
		
		return tv;  // quantized transformation array
	}

	// encode single transformation
	static unsigned int encode(const vector<unsigned int>& q, const unsigned int z[3])
		{ return q[0] * z[2] + q[1] * z[1] + q[2] * z[0] + q[3]; }

	// decode single transformation
	static vector<unsigned int> decode(const unsigned int& c, const unsigned int z[3])
	{
		unsigned int r;
		vector<unsigned int> d(4);
		d[0] = c / z[2];     r = c % z[2];
		d[1] = r / z[1];     r = r % z[1];
		d[2] = r / z[0];  d[3] = r % z[0];
		return d;
	}

	// quantize single encoded transformation
	// (decode, quantize at level l and re-encode)
	static unsigned int quant(const unsigned int& c, const unsigned int z[3], size_t l)
		{ vector<unsigned int> d = decode(c, z); for(size_t i=0;i<d.size();i++) d[i] >>= l; return encode(d , z); }

	// quantize vector of encoded transformations
	static vector<unsigned int> quant(const vector<unsigned int>& c,
		const unsigned int z[3], size_t l)
	{
		vector<unsigned int> q(c.size());
		for(size_t i=0; i<c.size(); i++) q[i] = quant(c[i], z, l);
		return q;
	}
};

typedef feature<double> feat;  // feature or similarity transformation

} // namespace hpm

#endif  // _FEATURE_HPP_