#pragma once
#include <cmath>
#include <initializer_list>
#include <algorithm>

struct Vec2i {
    int x = 0;
    int y = 0;

    Vec2i() = default;
    Vec2i(int _x, int _y) : x(_x), y(_y) {}
};

struct Vec2f {
    float x = 0.f;
    float y = 0.f;

    Vec2f() = default;
    Vec2f(float _x, float _y) : x(_x), y(_y) {}

    Vec2f operator+(const Vec2f& r) const { return Vec2f(x + r.x, y + r.y); }
    Vec2f operator-(const Vec2f& r) const { return Vec2f(x - r.x, y - r.y); }
    Vec2f operator*(float s) const { return Vec2f(x * s, y * s); }
    Vec2f operator/(float s) const { return Vec2f(x / s, y / s); }
};

struct Vec3f {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    Vec3f() = default;
    Vec3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    float norm() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3f& normalize() {
        float n = norm();
        if (n > 1e-12f) { x /= n; y /= n; z /= n; }
        return *this;
    }

    Vec3f operator+(const Vec3f& r) const { return Vec3f(x + r.x, y + r.y, z + r.z); }
    Vec3f operator-(const Vec3f& r) const { return Vec3f(x - r.x, y - r.y, z - r.z); }
    Vec3f operator*(float s) const { return Vec3f(x * s, y * s, z * s); }
    Vec3f operator/(float s) const { return Vec3f(x / s, y / s, z / s); }

    
    float operator*(const Vec3f& r) const { return x * r.x + y * r.y + z * r.z; }

    
    Vec3f operator^(const Vec3f& r) const {
        return Vec3f(
            y * r.z - z * r.y,
            z * r.x - x * r.z,
            x * r.y - y * r.x
        );
    }
};

struct Vec4f {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float w = 1.f;

    Vec4f() = default;
    Vec4f(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
};

struct Matrix {
    float m[4][4];

    Matrix(); 
    static Matrix identity();

    float* operator[](int i) { return m[i]; }
    const float* operator[](int i) const { return m[i]; }

    Matrix operator*(const Matrix& r) const;
};

Vec4f operator*(const Matrix& a, const Vec4f& v);
