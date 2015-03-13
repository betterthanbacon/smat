#include <smat/shape.h>
#include <base/util.h>
#include <base/assert.h>

SM_NAMESPACE_BEGIN

using namespace std;

string shape2str(const shape_t& shape)
{
	if (shape.z != 0) return format("(%lld,%lld,%lld)",(long long)shape.x,(long long)shape.y,(long long)shape.z);
	if (shape.y != 0) return format("(%lld,%lld)",(long long)shape.x,(long long)shape.y);
	return format("(%lld,)",(long long)shape.x);
}

coord_t fullstride(const shape_t& shape)
{
	return coord_t(shape.x == 0 ? 0 : 1,
	               shape.y == 0 ? 0 : shape.x,
	               shape.z == 0 ? 0 : shape.x*shape.y);
}

void slice_t::bind(isize_t dim)
{
	// "Bind" a slice to a specific dimension size, 
	// meaning negative indices get wrapped, and 
	// slice_end gets mapped to the actual dim size.
	if (first < 0) {
		first += dim;
		if (last <= 0) 
			last += dim;
	} else if (last < 0)
		last += dim;
	if (last == slice_end) 
		last = dim;
	SM_ASSERTMSG(first >= 0,"IndexError: Index out of range.\n");
	SM_ASSERTMSG(first <= last,"IndexError: Invalid slice.\n");
	SM_ASSERTMSG(last  <= dim,"IndexError: Index out of range.\n");
}

SM_NAMESPACE_END
