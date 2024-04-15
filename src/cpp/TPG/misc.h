#ifndef misc_h
#define misc_h
#include <cstring>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
//#include <bzlib.h>
#include <deque>
#include <chrono>
#include <sstream>
#include <any>
//#include <boost/iostreams/filtering_streambuf.hpp>
//#include <boost/iostreams/copy.hpp>
//#include <boost/iostreams/filter/gzip.hpp>

using namespace std;
using std::numeric_limits;
typedef double behaviourType;

#define NEARZERO 10e-12
#define EPSILON_SBB 1e-5
#define MAX_NCD 1.2
#define _MEAN_OUT_PROP 1.0
#define MIN_NIBBLE_VAL 0
#define MAX_NIBBLE_VAL 15
#define _TRAIN_PHASE 0
#define _VALIDATION_PHASE 1
#define _TEST_PHASE 2
#define _PLAY_PHASE 3
#define _NUM_PHASE 4

int compressedLength(char *);

inline double bound(double x, double m, double M) { return min(max(x, m), M); }
inline double discretize(double f,double min, double max, int steps){
   double d = round(((f - min)/(max - min))*(steps-1));
   return d>steps?steps-1:d;
}
void die(const char*, const char *, const int, const char *);
double EuclideanDistSqrd(double *, double *, int);
double EuclideanDistSqrd(std::vector < double > &, std::vector < double > &);
double EuclideanDistSqrdNorm(std::vector < double > &, std::vector < double > &);
double EuclideanDist(std::vector < double > &, std::vector < double > &);

inline bool fileExists(const char *fileName)
{
   ifstream infile(fileName);
   return infile.good();
}

int hammingDist(std::vector < int > &, std::vector < int > &);

inline bool isEqual(double x, double y)
{ return fabs(x - y) < NEARZERO; }

inline bool isEqual(double x, double y, double e)
{ return fabs(x - y) < e; }

inline bool isEqual(unsigned char x, unsigned char y)
{ return fabs(x - y) < 0; }

bool isEqual(std::vector < int > &, std::vector < int > &);
bool isEqual(std::vector < double > &, std::vector < double > &, double);
template<class T>
bool isEqual(std::vector < T > &, std::vector < T > &, double);
inline bool isGreater(double x, double y, double e, bool orEqual = false) { return orEqual ? x > y || fabs(x - y) < e : x > y && fabs(x - y) > e; }
inline bool isLess(double x, double y, double e, bool orEqual = false) { return orEqual ? x < y || fabs(x - y) < e : x < y && fabs(x - y) > e; }

struct modesRecord{
   set < long > activeProgramIds;
   set < long > activeTeamIds;
   string behaviourString;
   long effectiveInstructionsTotal;
   double runTimeComplexityIns;
   modesRecord(){ effectiveInstructionsTotal = 0; runTimeComplexityIns = 0;}
};
//double normalizedCompressionDistance(std::vector<int>&v1,std::vector<int>&v2);
//double normalizedCompressionDistance(string&v1, string&v2);

struct noveltyDescriptor {
   double novelty;
   std::vector < int > profile;
   std::vector < long > profileLong;
} ;

struct phyloRecord{
   std::vector < long > adj;
   set < long > ancestorIds;
   string behaviourString;
   long gtime;
   long dtime;
   string fitnessBin;
   double fitness;
   bool root;
   long numActiveFeatures;
   long numActivePrograms;
   long numActiveTeams;
   long numEffectiveInstructions;
   phyloRecord(){ gtime = -1; dtime = -1; fitnessBin = ""; fitness = 0; root = true; }
};

inline void getAncestorIds(map <long, phyloRecord > &phyloGraph, set < long > &a, long id) {
   a.insert(phyloGraph[id].ancestorIds.begin(), phyloGraph[id].ancestorIds.end());
   for (auto it = phyloGraph[id].ancestorIds.begin(); it != phyloGraph[id].ancestorIds.end(); it++)
      getAncestorIds(phyloGraph, a, *it);
}

int readMap(string, map < string, string > &);
void ReadParameters(string file_name, std::unordered_map<string, std::any> &params);
inline double sigmoid(double x, double m) { return 1 / (1 + exp(-(m*x))); }
double stdDev(std::vector<double>);
int stringToInt(string);
long stringToLong(string);
double stringToDouble(string);

inline double sas(double s1,double s2, double a){
   return sqrt(pow(s1,2) + pow(s2,2) - (2*s1*s2*cos(a*(3.14159265/180.0))));
}
std::vector<string> &splitString(const string &s, char delim, std::vector<string> &elems);
std::vector<string> splitString(const string &s, char delim);

template < class ptype > struct lessThan : public binary_function < ptype *, ptype *, bool >
{
   bool operator() (ptype *lhs, ptype *rhs) { return lhs->key() < rhs->key(); }
};
template < class ptype > struct greaterThan : public binary_function < ptype *, ptype *, bool >
{
   bool operator() (ptype *lhs, ptype *rhs) { return lhs->key() > rhs->key(); }
};
template < class vtype > string vecToStr(std::vector < vtype > &v)
{ ostringstream oss; //oss.precision(numeric_limits<double>::digits10+1);
   for(size_t i = 0; i < v.size(); i++) {
      oss << v[i];
      if (i < v.size() - 1)
         oss << " ";
   }
   return oss.str();
}
template < class vtype > string setToStr(set < vtype > &s)
{ ostringstream oss; //oss.precision(numeric_limits<double>::digits10+1);
   for(auto it = s.begin(); it != s.end(); it++) {
      oss << *it;
      if (next(it) != s.end())
         oss << " ";
   }
   return oss.str();
}
template < class vtype > string vecToStrNoSpace(std::vector < vtype > &v)
{ ostringstream oss; for(size_t i = 0; i < v.size(); i++) { oss << v[i]; } return oss.str(); }

double vecMedian(std::vector<double>);
int vecMedian(std::vector<int>);
double vecMean(std::vector<double>);
double vecMean(std::vector<int>);

//string compressString(std::string& data);
//string decompressString(std::string& data);

// Function to generate power set PS of given set S
inline void findPowerSet(std::vector<int> const &S, std::vector<int> &set, std::vector < std::vector <int> > &PS, size_t n, size_t minSubsetSize)
{
   // if we have considered all elements
   if (n == 0)
   {
      if (set.size() >= minSubsetSize)
         PS.push_back(set);
      return;
   }
   // consider nth element
   set.push_back(S[n - 1]);
   findPowerSet(S, set, PS, n - 1, minSubsetSize);
   // or don't consider nth element
   set.pop_back();
   findPowerSet(S, set, PS, n - 1, minSubsetSize);
}

inline std::vector < std::vector < int > > powerSet(size_t n) {
   std::vector <int> S(n);
   std::iota(S.begin(), S.end(), 0);
   std::vector <int> tmpSet;
   std::vector < std::vector < int > > PS;
   findPowerSet(S, tmpSet, PS, n, 1);
   return PS;
}

// class CSVReader
// {
//    string fileName;
//    string delimiter;
//    int dim;
//    public:
//    CSVReader(string fname, int d, string delm = " "){
//       fileName = fname;
//       delimiter = delm;
//       dim = d;
//    }
//    std::vector < std::vector <double> > getData()
//    {
//       ifstream file(fileName);
//       std::vector < std::vector <double> > dataVec;
//       string line = "";
//       while (getline(file, line))
//       {
//          std::vector<double> doubleValues(dim);//features
//          doubleValues[0] = stod(line.c_str());
//          dataVec.push_back(doubleValues);
//       }
//       file.close();
//       return dataVec;
//    }
// };

#endif
