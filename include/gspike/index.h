// Index type used for every size, stride and offset in gspike.
//
// The width is selected at configure time with the CMake option
// GSPIKE_INDEX_TYPE (int32 by default, or int64), which defines
// GSPIKE_INDEX_64 for the int64 case.  The setting is part of the ABI: the
// library and the code using it must be compiled with the same choice, which
// CMake does automatically for targets that link gspike.
//
// The public header and the device kernels are both parameterized on
// gspike::index_t, so all index arithmetic is done in the configured type.
#pragma once

#include <cstdint>

namespace gspike {

#if defined(GSPIKE_INDEX_64)
    using index_t = std::int64_t;
    inline constexpr bool index_64bit = true;
#else
    using index_t = std::int32_t;
    inline constexpr bool index_64bit = false;
#endif

} // namespace gspike
