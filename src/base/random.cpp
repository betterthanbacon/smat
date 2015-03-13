#include <base/random.h>
#include <random>
#include <memory>

using namespace std;

SM_NAMESPACE_BEGIN

size_t g_rand_seed = 0;
static mt19937 g_rand_gen;
static uniform_real_distribution<double> g_uniform_dbl;
static uniform_real_distribution<float> g_uniform_flt;
static uniform_int_distribution<int> g_uniform_int;
static normal_distribution<double> g_normal_dbl;
static normal_distribution<float> g_normal_flt;

BASE_EXPORT void    set_rand_seed(size_t seed)  { g_rand_seed = seed; g_rand_gen.seed((unsigned long)seed); }
BASE_EXPORT size_t  bump_rand_seed()            { return g_rand_seed *= 1234; }
BASE_EXPORT double  rand_double()  { return g_uniform_dbl(g_rand_gen); }
BASE_EXPORT float   rand_float()   { return g_uniform_flt(g_rand_gen); }
BASE_EXPORT int     rand_int()     { return g_uniform_int(g_rand_gen); }
BASE_EXPORT double  randn_double() { return g_normal_dbl(g_rand_gen);  }
BASE_EXPORT float   randn_float()  { return g_normal_flt(g_rand_gen);  }

SM_NAMESPACE_END
