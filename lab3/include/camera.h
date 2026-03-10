#pragma once
#include "geometry.h"

class Camera {
public:
    Camera(const Vec3f& eye,
        const Vec3f& center,
        const Vec3f& up,
        float fovY_deg,
        float aspect,
        float zNear,
        float zFar)
        : m_eye(eye)
        , m_center(center)
        , m_up(up)
        , m_fovY(fovY_deg)
        , m_aspect(aspect)
        , m_zNear(zNear)
        , m_zFar(zFar)
    {
    }

    Matrix viewMatrix() const;
    Matrix projectionMatrix() const;

    const Vec3f& position() const { return m_eye; }

private:
    Vec3f m_eye;
    Vec3f m_center;
    Vec3f m_up;

    float m_fovY = 60.f;
    float m_aspect = 1.f;
    float m_zNear = 0.1f;
    float m_zFar = 100.f;
};
