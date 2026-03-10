#include "geometry.h"

template<>
template<>
Vec3<int>::Vec3(const Vec3<float>& v)
    : x(static_cast<int>(v.x + (v.x >= 0.f ? 0.5f : -0.5f))),
    y(static_cast<int>(v.y + (v.y >= 0.f ? 0.5f : -0.5f))),
    z(static_cast<int>(v.z + (v.z >= 0.f ? 0.5f : -0.5f))) {
}

template<>
template<>
Vec3<float>::Vec3(const Vec3<int>& v)
    : x(static_cast<float>(v.x)),
    y(static_cast<float>(v.y)),
    z(static_cast<float>(v.z)) {
}
