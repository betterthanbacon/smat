#ifndef __SM_RANGE_H__
#define __SM_RANGE_H__

#include <base/config.h>
#ifdef _MSC_VER
#include <xutility>  // for std::input_iterator_tag
#else
#include <iterator>
#endif

SM_NAMESPACE_BEGIN

template <typename T>
class _range {
public:
	class const_iterator {
	public:
		SM_INLINE const_iterator() {}
		SM_INLINE const_iterator(T pos, T step = 1): _pos(pos), _step(step) {}
		SM_INLINE const_iterator& operator++() { _pos += _step; return *this; }
		SM_INLINE const_iterator& operator--() { _pos -= _step; return *this; }
		SM_INLINE bool operator==(const const_iterator& other) const { return _pos == other._pos; }
		SM_INLINE bool operator!=(const const_iterator& other) const { return _pos != other._pos; }
		SM_INLINE const T& operator*() const { return _pos; }

		typedef std::input_iterator_tag iterator_category;
		typedef T value_type;
		typedef void difference_type;
		typedef const T* pointer;
		typedef const T& reference;
	private:
		T _pos;
		T _step;
	};

	SM_INLINE _range(T begin, T end, T step) : _begin(begin), _end(end), _step(step) {}
	SM_INLINE bool operator==(const _range &other) const { return _begin == other._begin && _end == other._end; }
	SM_INLINE const_iterator begin() const { return const_iterator(_begin,_step); }
	SM_INLINE const_iterator end()   const { return const_iterator(_end,_step); }

private:
	T _begin,_end,_step;
};

template <typename T> _range<T> range(T count)                { return _range<T>(0,count,1); }
template <typename T> _range<T> range(T begin, T end)         { return _range<T>(begin,end,1); }
template <typename T> _range<T> range(T begin, T end, T step) { return _range<T>(begin,end,step); }

SM_NAMESPACE_END

#endif // __SM_RANGE_H__
