#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <vector>

namespace hpm {

//------------------------------------------------------------
// implode a (vector/array) container of containers into a
// flat container
// see double_array2array above

template<class C>
typename C::value_type implode(const C& in)
{
	// count elements
	int n = 0;
	for(size_t i=0; i<in.size(); i++) n += in[i].size();

	// resize output
	typename C::value_type out(n);

	// copy elements
	for(size_t i=0, c=0; i<in.size(); i++)
		for(size_t j=0; j<in[i].size(); j++, c++)
			out[c] = in[i][j];

	return out;
}

//---------------------------------------------------------------------
// get "columns" out of a vector of vectors

template <class T>
std::vector<std::vector<T> > get_columns(const std::vector<std::vector<T> >& a, const std::vector<size_t>& columns)
{
	std::vector<std::vector<T> > col(a.size());
	for (size_t i = 0 ; i < a.size(); i++){
		const std::vector<T>& ta=a[i];
		std::vector<T> this_col(columns.size());
		for (size_t j=0; j < columns.size(); j++)
			this_col[j] = ta[ columns[j] ];
		col[i] = this_col;
	}
	return col;
}

//---------------------------------------------------------------------
// flip the order of elements of a vector

template<class C>
std::vector<C> flip(const std::vector<C>& a)
{
	std::vector<C> b;
	for(int i=a.size()-1;i>=0;i--)
		b.push_back(i);
	return b;
}

//---------------------------------------------------------------------
// sort a vector and return the re-ordered indices

template<class C>
C sort(const C& a, std::vector<size_t> &ind, bool descending = false)
{
	ind.resize(a.size());
	for(size_t i=0;i<a.size();i++)
		ind[i] = i;

	C c(a);
	long gap = static_cast<long>(a.size()) + 1;
	bool more = false;

	while (gap > 1  || more) {
		if (gap > 1)
			gap = static_cast<long>(gap / 1.3);

		if (gap == 9 || gap == 10)
			gap = 11;

		more = false;
		for (size_t i = 0; i < a.size() - gap; i++) {
			if ((c[i] > c[i + gap])) {
				// if the items are not in order, swap them
				std::swap(c[i],c[i + gap]);
				std::swap(ind[i],ind[i+gap]); // keep index too
				more = true;
			}
		}
	}

	if (descending)
	{
		c = flip(c);
		ind = flip(ind);
	}
	return c;
}

//---------------------------------------------------------------------
// sort a vector according to the order of another

template<class C, class D>
C order_by(const C& a, const D& b, bool descending = false)
{
	// get the order by sorting b
	std::vector<size_t> ind;
	sort(b, ind, descending);

	// then, apply this order to a and yield c
	C c(a.size());
	for(size_t i=0; i<a.size(); i++) c[i] = a[ind[i]];
	return c;
}

//------------------------------------------------------------------------------------------------
// intersection of two vectors which are both unique and sorted, 
// returns common elements and indices to them

template<class T> inline
size_t intersect_us(const std::vector<T>& a, const std::vector<T>& b,
					std::vector<T>& common, std::vector<size_t>& ia, std::vector<size_t> &ib)
{
	//! TODO, add also frequencies to the intersection
	size_t size_a = a.size();
	size_t size_b = b.size();

	size_t i,j;
	i=j=0;

	std::vector<size_t> iav,ibv;

	while(i<size_a && j<size_b)
	{
		int t = compare( a[i] , b[j] );
		if( t == 0 ) //add a correspondce
		{
			iav.push_back(i);
			ibv.push_back(j);
			i++;
			j++;
		}
		else if( t == 1) //increase the index of b, because the current element of a is higher than the current element of b
			j++;
		else
			i++; //increase the index of a
	}

	size_t n_c = iav.size();
	common.resize(n_c);
	ia.resize(n_c);
	ib.resize(n_c);

	for(size_t k=0;k<n_c;k++)
	{
		ia[k] = iav[k];
		ib[k] = ibv[k];
		common[k] = a[ia[k]];
	}

	return ia.size();
}

//------------------------------------------------------------------------------------------------
// intersection of two vectors which are both unique and sorted, using frequencies of elements
// returns common elements and indices to them

template<class T> inline
size_t intersect_us(const std::vector<T>& a, const std::vector<T>& b,
					const std::vector<size_t>& fa, const std::vector<size_t>& fb,
					std::vector<T>& common, std::vector<size_t>& ia, std::vector<size_t>& ib)
{
	intersect_us(a,b,common,ia,ib);
	size_t in =0;
	for(size_t i =0;i<ia.size();i++)
		in+= std::min( fa[ia[i]], fb[ib[i]] );

return in;
}

//------------------------------------------------------------------------------------------------
// intersection of two vectors
// returns common elements and indices to them

template<class T>
size_t intersect(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& common,
				std::vector<std::vector<size_t> >& ia, std::vector<std::vector<size_t> >& ib)
{
	std::vector<size_t> fa,fb;
	std::vector<std::vector<size_t> > inda,indb;
	std::vector<T> a_us = unique_count(a,fa,inda);
	std::vector<T> b_us = unique_count(b,fb,indb);

	std::vector<size_t> iia,iib;
	size_t intersections  = intersect_us(a_us,b_us,fa,fb, common, iia, iib);

	ia.resize(iia.size());
	ib.resize(iib.size());

	for(size_t i=0;i<ia.size();i++) //find index to the initial non-unique vectors
	{
		ia[i] = inda[iia[i]];
		ib[i] = indb[iib[i]];
	}

return intersections;
}

//compare two elements, returns 0 if equal, 1 if a>b and -1 if a<b

template <class T>
inline int compare(const T &a,const T &b)
{
	if(a>b)
		return 1; //a higher than b
	else if(a<b)
		return -1;//b higher than a

	return 0;//a and b are equal
}

//------------------------------------------------------------------------------------------------
// find unique elements of a vector, their frequency and re-order indices

template<class T>
std::vector<T> unique_count(const std::vector<T>& a, std::vector<size_t>& f, std::vector<std::vector<size_t> >& index)
{
	std::vector<size_t> ind;
	std::vector<T> s_a(a);
	radixsort(s_a,ind); //sort the array

	std::vector<std::vector<size_t> > ind_temp;
	std::vector<T> un_a =  unique_count_core(s_a,f,ind_temp); //unique_count to the sorted vector

	index.resize(ind_temp.size());  //find index from the final unique vector to the initial non_unique and non_sorted vector
	for(size_t i =0;i<ind_temp.size();i++)
	{
		index[i].resize(ind_temp[i].size());
		for(size_t j=0;j<ind_temp[i].size();j++)
			index[i][j] = ind[ind_temp[i][j]];
	}

return un_a;
}

//------------------------------------------------------------------------------------------------
// find unique elements of a vector (core function)

template<class T>
std::vector<T> unique_count_core(const std::vector<T>& s_a, std::vector<size_t>& f, 
								 std::vector<std::vector<size_t> >& index, bool option = true)
{
	std::vector<T> unique_a(0);
	if(s_a.size()==0)
		return unique_a;
	//create a mask to mark where there is a d-dimensional indice appearing for the first time, in order to keep only unique indices
	std::vector<size_t> mask(s_a.size(),(size_t)0);
	mask[0]=1;
	// fill in the mask for the unique indices, mask also includes the occurences
	size_t last = 0;
	for(size_t i=1;i<s_a.size();i++)
		if( compare( s_a[i] , s_a[i-1]) == 1) // if a new element is found
		{
			mask[i]=1;  // then mark it and add 1
			last = i; // and keep its index
		}
		else
			mask[last]++; // if its not a new value then add one at its last occurence

	unique_a.clear();
	f.clear();
	std::vector<size_t> unique;
	for(size_t i=0;i<mask.size();i++)
		if(mask[i]>0)
		{
			unique.push_back(i);
			unique_a.push_back(s_a[i]);
			f.push_back(mask[i]);
		}

	//keep index for the new vector (how to go back to the non unique vector
	if(option)
	{
		index.resize(unique_a.size());
		size_t k=0;
		size_t m=0;
		for(size_t i=0;i<index.size();i++)
		{
			index[i].resize(mask[unique[m]]);
			for(size_t j=0;j<index[i].size();j++)
			{
				index[i][j] = k++;
			}
			m++;
		}
	}
	else
		index.resize(0);

return unique_a;
}

//-----------------------------------------------------------------------------------------------
// sort a vector using the radixsort algorithm and return the re-ordered indices

template<class T> inline
void radix(size_t b, std::vector<T> &source, std::vector<T> &dest, std::vector<size_t> &ind)
{
	std::vector<size_t> t_ind(ind);
	size_t N = source.size();
	T count[256];
	T index[256];
	for(size_t i=0;i<256;i++) count[i]=0;

	for ( size_t i=0; i<(size_t)N; i++ )   //count occurences of  each possible element
		count[((source[i])>>(b*8))&0xff]++;

	index[0]=0;
	for ( size_t i=1; i<256; i++ )  //build a index-list for each possible element
		index[i]=index[i-1]+count[i-1];
	for ( size_t i=0; i<(size_t)N; i++ )  //place elements in another array with the sorted order
	{
		size_t pos = (size_t)( ((source[i])>>(b*8))&0xff);
		dest[(size_t)index[pos]] = (T)source[i];
		ind[(size_t)index[pos]] = t_ind[i];
		index[pos]++;
	}
}

//-------------------------------------------------------------------------------------
// sort a vector using the radixsort algorithm and return the re-ordered indices

template<class T> inline
void radixsort(std::vector<T> &source, std::vector<size_t> &ind)
{
	for(size_t i=0;i<source.size();i++)
		ind.push_back(i);

	size_t nb = sizeof(source[0]) / sizeof(char);  //number of bytes used
	std::vector<T> temp(source.size()); //temporary array

	for(size_t i=0; i<nb/2; i++) // sort based on each byte starting from the LSD
	{
		radix (i*2, source, temp, ind);
		radix (i*2+1, temp, source, ind);
	}
}

//------------------------------------------------------------------------
// true if vectors are different

template<class T1, class T2> inline
bool neq(const std::vector<T1>& a, const std::vector<T2>& b)
{
	for(size_t i=0, len=a.size(); i < len; i++)
		if(a[i]!=b[i]) return true;
	return false;
}

}  // namespace hpm

#endif  //_UTIL_HPP_
