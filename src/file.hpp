#ifndef _FILE_HPP_
#define _FILE_HPP_

#include <fstream>
#include <iostream>
#include <string>

//---------------------------------------------------------
template<class T>
void load_stream(T& a, std::ifstream& ifstr)
{
	T b;
    ifstr.read(reinterpret_cast<char *>(&b), sizeof(T));
	a = b;
}

//---------------------------------------------------------
template<class T>
void load_stream(std::vector<T>& a, std::ifstream& ifstr)
{
	double sz;
	ifstr.read(reinterpret_cast<char *>(&sz), sizeof(sz));
	a.resize((size_t)sz);

	for(size_t i=0;i<a.size();i++)
		load_stream(a[i],ifstr);
}

//---------------------------------------------------------
template<class T>
void save_stream(const T& a, std::ofstream& ofstr)
{
	T b = a;
	ofstr.write(reinterpret_cast<char *>(&b), sizeof(T));
}

//---------------------------------------------------------
template<class T>
void save_stream(const std::vector<T>& a, std::ofstream& ofstr)
{
	double sz = (double)a.size();
	ofstr.write(reinterpret_cast<char *>(&sz), sizeof(sz));

	for(size_t i=0;i<a.size();i++)
		save_stream(a[i],ofstr);
}

//---------------------------------------------------------
template<class T>
void load_bin(T& a, const std::string &filename)
{
	std::ifstream file(filename.c_str(), std::ios::binary);
	load_stream(a,file);
	file.close();
}

//---------------------------------------------------------
template<class T>
void save_bin(const T& a, const std::string &filename)
{
	std::ofstream file(filename.c_str(), std::ios::binary);
	save_stream(a,file);
	file.close();
}

#endif  // _FILE_HPP_
