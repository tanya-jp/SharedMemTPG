#include "misc.h"

///***********************************************************************************************************/
//int compressedLength(char * source){
//   ostringstream oss;
//   int blockSize100k = 9;
//   int verbosity = 0;
//   int workFactor = 30; // 0 = USE THE DEFAULT VALUE
//   unsigned int sourceLength = strlen(source);
//   unsigned int destLength = 1.01 * sourceLength + 600;    // Official formula, Big enough to hold output.  Will change to real size.
//   char *dest = (char*)malloc(destLength);
//   int returnCode = BZ2_bzBuffToBuffCompress( dest, &destLength, source, sourceLength, blockSize100k, verbosity, workFactor );
//
//   if (returnCode == BZ_OK)
//   {
//      free(dest);
//      return destLength;
//   }
//   else
//   {
//      free(dest);
//      cout << " Can't get compressed length. " << "Error code:" << returnCode;
//      return returnCode;
//   }
//}

///***********************************************************************************************************/
//
//string compressString(std::string& data)
//{
//   namespace bio = boost::iostreams;
//
//   std::stringstream compressed;
//   std::stringstream origin(data);
//
//   bio::filtering_streambuf<bio::input> out;
//   out.push(bio::gzip_compressor(bio::gzip_params(bio::gzip::best_compression)));
//   out.push(origin);
//   bio::copy(out, compressed);
//
//   //return compressed.str();
//   string decomp = compressed.str();
//   return decompressString(decomp);
//}
//
///***********************************************************************************************************/
//
//string decompressString(std::string& data)
//{
//   namespace bio = boost::iostreams;
//
//   std::stringstream compressed(data);
//   std::stringstream decompressed;
//
//   bio::filtering_streambuf<bio::input> out;
//   out.push(bio::gzip_decompressor());
//   out.push(compressed);
//   bio::copy(out, decompressed);
//
//   return decompressed.str();
//}
//
/***********************************************************************************************************/

void die(const char *file, 
      const char *func,
      const int line,
      const char *msg)
{
   cerr << "error in file " << string(file) << " function " << string(func);
   cerr << " line " << line << ": " << string(msg) << "... exiting" << endl;
   abort();
}

/***********************************************************************************************************/
double EuclideanDistSqrd(double *x,
      double *y,
      int dim)
{
   double dist = 0;

   for(int i = 0; i < dim; i++)
      dist += (x[i] - y[i]) * (x[i] - y[i]);

   return dist;
}

/***********************************************************************************************************/
double EuclideanDistSqrd(vector < double > &x,
      vector < double > &y)
{
   double dist = 0;
   vector < double > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end();
         xiter != enditer; xiter++, yiter++)
      dist += (*xiter - *yiter) * (*xiter - *yiter);

   return dist;
}

/***********************************************************************************************************/
double EuclideanDistSqrdNorm(vector < double > &x,
      vector < double > &y)
{
   double dist = 0;
   int numFeatures = 0;
   vector < double > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end();
         xiter != enditer; xiter++, yiter++){
      dist += (*xiter - *yiter) * (*xiter - *yiter);
      numFeatures++;
   }
   return dist/numFeatures;
}

/***********************************************************************************************************/
double EuclideanDist(vector < double > &x,
      vector < double > &y)
{
   double dist = 0;
   vector < double > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end(); xiter != enditer; xiter++, yiter++)
      dist += (*xiter - *yiter) * (*xiter - *yiter);

   return (double)sqrt(dist);
}

/***********************************************************************************************************/
int hammingDist(vector < int > &x, 
      vector < int > &y)
{
   int dist = 0;
   vector < int > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end();
         xiter != enditer; xiter++, yiter++)
      dist += *xiter == *yiter? 0 : 1;

   return dist;
}

/***********************************************************************************************************/
bool isEqual(vector < int > &x, 
      vector < int > &y)
{
   if(x.size() != y.size()) return false;

   vector < int > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end();
         xiter != enditer; xiter++, yiter++)
      if(*xiter != *yiter) return false;

   return true;
}

/***********************************************************************************************************/
bool isEqual(vector < double > &x,
      vector < double > &y,
      double e)
{
   if(x.size() != y.size()) return false;

   vector < double > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end();
         xiter != enditer; xiter++, yiter++)
      if(isEqual(*xiter, *yiter, e) == false) return false;

   return true;
}

/***********************************************************************************************************/
   template<class T>
bool isEqual(vector < T > &x, vector < T > &y, double e)
{
   if(x.size() != y.size()) return false;

   typename vector < T > :: iterator xiter, yiter, enditer;

   for(xiter = x.begin(), yiter = y.begin(), enditer = x.end();
         xiter != enditer; xiter++, yiter++)
      if(isEqual(*xiter, *yiter, e) == false) return false;

   return true;
}

///***********************************************************************************************************/
//double normalizedCompressionDistance(vector<int>&v1,vector<int>&v2){
//   if (v1 == v2)
//      return 0;
//   ostringstream o;
//   o << vecToStrNoSpace(v1);
//   int Zx = compressedLength((char*) o.str().c_str());
//   o.str("");
//   o << vecToStrNoSpace(v2);
//   int Zy = compressedLength((char*) o.str().c_str());
//   o.str("");
//   o << vecToStrNoSpace(v1) << vecToStrNoSpace(v2);
//   int Zxy = compressedLength((char*) o.str().c_str());
//   o.str("");
//   int nom = Zxy-min(Zx,Zy);
//   int denom = max(Zx,Zy);
//   return ((double)nom/denom)/MAX_NCD;
//}
//
///***********************************************************************************************************/
//double normalizedCompressionDistance(string&v1, string&v2){
//   if (v1 == v2)
//      return 0;
//   int Zx = compressedLength((char*) v1.c_str());
//   int Zy = compressedLength((char*) v2.c_str());
//   int Zxy = compressedLength((char*) (v1+v2).c_str());
//   int nom = Zxy-min(Zx,Zy);
//   int denom = max(Zx,Zy);
//   return ((double)nom/denom)/MAX_NCD;
//}

/***********************************************************************************************************/
int readMap(string fileName, 
      map < string, string > &args)
{
   ostringstream o;
   o << "cannot open map file: " << fileName;
   int pairs = 0;

   ifstream infile(fileName.c_str(), ios::in);

   if(!infile)
      die(__FILE__, __FUNCTION__, __LINE__, o.str().c_str());

   do
   {
      string key, value;

      if(infile) infile >> key; else break;
      if(infile) infile >> value; else break;

      args.insert(map < string, string > :: value_type(key, value));
      pairs++;

   } while(true);

   infile.close();

   return pairs;
}

///***********************************************************************************************************/
//void ReadParameters(string file_name, std::unordered_map<string, std::any> &params)
//{
//   std::ifstream infile(file_name);
//   string oneline;
//   vector < string > outcome_fields;
//   while (std::getline(infile, oneline)) {
//	   if (oneline.find('#') != std::string::npos) continue;  // skip comment lines
//	   splitString(oneline,' ',outcome_fields);
//	   //cout << "param: " << vecToStr(outcome_fields) << endl;
//	   if (outcome_fields[1].find('.') != std::string::npos) // found double parameter
//             params[outcome_fields[0]] = stringToDouble(outcome_fields[1]);
//	   else  // found int parameter
//	     params[outcome_fields[0]] = stringToInt(outcome_fields[1]);
//	   if (outcome_fields[0].at(0) == '_' && outcome_fields[1] == '1')
//		   _ops[outcome_fields[0]
//   }
//}

/***********************************************************************************************************/
int stringToInt(string s)
{
   istringstream buffer(s);

   int i;

   buffer >> i;

   return i;
}

/***********************************************************************************************************/
long stringToLong(string s)
{
   istringstream buffer(s);

   long l;

   buffer >> l;

   return l;
}

/***********************************************************************************************************/
double stringToDouble(string s)
{
   istringstream buffer(s);

   double d;

   buffer >> d;

   return d;
}

/***********************************************************************************************************/
double stdDev(vector<double> vec){
   double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
   double mean = sum / vec.size();
   vector<double> diff(vec.size());
   transform(vec.begin(), vec.end(), diff.begin(),
         bind2nd(std::minus<double>(), mean));
   double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
   double stdev = sqrt(sq_sum / (vec.size()-1));
   return stdev;
}

/***********************************************************************************************************/
vector<string> &splitString(const string &s, char delim, vector<string> &elems) {
   elems.clear();
   stringstream ss(s);
   string item;
   while (getline(ss, item, delim)) {
      elems.push_back(item);
   }
   return elems;
}

/***********************************************************************************************************/
vector<string> splitString(const string &s, char delim) {
   vector<string> elems;
   splitString(s, delim, elems);
   return elems;
}

/***********************************************************************************************************/
double vecMedian(vector<double> vec)
{
   typedef vector<double>::size_type vec_sz;

   vec_sz size = vec.size();

   if (size == 0)
      die(__FILE__, __FUNCTION__, __LINE__, "trying to get median of empty vector");

   sort(vec.begin(), vec.end());

   vec_sz mid = size/2;

   return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

/***********************************************************************************************************/
int vecMedian(vector<int> vec)
{
   typedef vector<int>::size_type vec_sz;

   vec_sz size = vec.size();

   if (size == 0)
      die(__FILE__, __FUNCTION__, __LINE__, "trying to get median of empty vector");

   sort(vec.begin(), vec.end());

   vec_sz mid = size/2;

   return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

/***********************************************************************************************************/
double vecMean(vector<double> vec)
{
   typedef vector<double>::size_type vec_sz;

   vec_sz size = vec.size();

   if (size == 0)
      return 0.0;

   double sum = accumulate(vec.begin(), vec.end(), 0.0);

   return sum/size;
}

/***********************************************************************************************************/
double vecMean(vector<int> vec)
{
   typedef vector<int>::size_type vec_sz;

   vec_sz size = vec.size();

   if (size == 0)
      return 0.0;

   double sum = accumulate(vec.begin(), vec.end(), 0.0);

   return sum/size;
}

