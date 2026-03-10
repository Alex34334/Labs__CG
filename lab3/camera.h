#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "geometry.h"

class Camera {
public:
    Camera(const Vec3f& position = Vec3f(5.f, 2.f, 5.f),
        const Vec3f& target = Vec3f(0.f, 0.f, 0.f),
        const Vec3f& up = Vec3f(0.f, 1.f, 0.f),
        float fovYDeg = 20.f,
        float aspect = 1.f,
        float zNear = 0.1f,
        float zFar = 100.f);

    void setPosition(const Vec3f& p);
    void setTarget(const Vec3f& t);
    void setUp(const Vec3f& u);
    void setPerspective(float fovYDeg, float aspect, float zNear, float zFar);

    Matrix viewMatrix() const;
    Matrix projectionMatrix() const;

    const Vec3f& position() const { return m_position; }
    const Vec3f& target()   const { return m_target; }
    const Vec3f& up()       const { return m_up; }

private:
    Vec3f m_position;
    Vec3f m_target;
    Vec3f m_up;
    float m_fovY;
    float m_aspect;
    float m_zNear;
    float m_zFar;
};

#endif
