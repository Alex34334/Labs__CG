#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>

template <class t>
struct Vec2 {
    t x, y;
    Vec2() : x(t()), y(t()) {}
    Vec2(t _x, t _y) : x(_x), y(_y) {}
    t& operator[](int i) { return (i == 0 ? x : y); }
    const t& operator[](int i) const { return (i == 0 ? x : y); }

    Vec2<t> operator+(const Vec2<t>& v) const { return Vec2<t>(x + v.x, y + v.y); }
    Vec2<t> operator-(const Vec2<t>& v) const { return Vec2<t>(x - v.x, y - v.y); }
    Vec2<t> operator*(float f) const { return Vec2<t>(x * f, y * f); }
};

typedef Vec2<float> Vec2f;
typedef Vec2<int>   Vec2i;

template <class t>
struct Vec3 {
    t x, y, z;

    Vec3() : x(t()), y(t()), z(t()) {}
    Vec3(t _x, t _y, t _z) : x(_x), y(_y), z(_z) {}

    t& operator[](int i) { return (i == 0 ? x : (i == 1 ? y : z)); }
    const t& operator[](int i) const { return (i == 0 ? x : (i == 1 ? y : z)); }

    template <class u>
    Vec3(const Vec3<u>& v) : x(static_cast<t>(v.x)), y(static_cast<t>(v.y)), z(static_cast<t>(v.z)) {}

    Vec3<t> operator+(const Vec3<t>& v) const { return Vec3<t>(x + v.x, y + v.y, z + v.z); }
    Vec3<t> operator-(const Vec3<t>& v) const { return Vec3<t>(x - v.x, y - v.y, z - v.z); }
    Vec3<t> operator*(float f) const { return Vec3<t>(x * f, y * f, z * f); }

    t operator*(const Vec3<t>& v) const { return x * v.x + y * v.y + z * v.z; }

    Vec3<t> operator^(const Vec3<t>& v) const {
        return Vec3<t>(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    float norm() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3<t>& normalize() {
        float n = norm();
        if (n > 1e-6f) {
            x /= n;
            y /= n;
            z /= n;
        }
        return *this;
    }
};

typedef Vec3<float> Vec3f;
typedef Vec3<int>   Vec3i;

struct Vec4f {
    float x, y, z, w;
    Vec4f() : x(0.f), y(0.f), z(0.f), w(1.f) {}
    Vec4f(float _x, float _y, float _z, float _w = 1.f) : x(_x), y(_y), z(_z), w(_w) {}
};

struct Matrix {
    float m[4][4];

    Matrix() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = 0.f;
    }

    static Matrix identity(int dim = 4) {
        Matrix r;
        for (int i = 0; i < dim && i < 4; i++) r.m[i][i] = 1.f;
        return r;
    }

    float* operator[](int i) { return m[i]; }
    const float* operator[](int i) const { return m[i]; }

    Matrix operator*(const Matrix& o) const {
        Matrix r;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++)
                    r.m[i][j] += m[i][k] * o.m[k][j];
        return r;
    }

    Vec4f operator*(const Vec4f& v) const {
        Vec4f r;
        r.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w;
        r.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w;
        r.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w;
        r.w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w;
        return r;
    }
};

#endif
