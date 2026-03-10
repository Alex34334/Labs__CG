#include <cmath>
#include "camera.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Camera::Camera(const Vec3f& position,
    const Vec3f& target,
    const Vec3f& up,
    float fovYDeg,
    float aspect,
    float zNear,
    float zFar)
    : m_position(position)
    , m_target(target)
    , m_up(up)
    , m_fovY(fovYDeg)
    , m_aspect(aspect)
    , m_zNear(zNear)
    , m_zFar(zFar) {
}

void Camera::setPosition(const Vec3f& p) { m_position = p; }
void Camera::setTarget(const Vec3f& t) { m_target = t; }
void Camera::setUp(const Vec3f& u) { m_up = u; }

void Camera::setPerspective(float fovYDeg, float aspect, float zNear, float zFar) {
    m_fovY = fovYDeg;
    m_aspect = aspect;
    m_zNear = zNear;
    m_zFar = zFar;
}

Matrix Camera::viewMatrix() const {
    Vec3f z = (m_position - m_target).normalize();
    Vec3f x = (m_up ^ z).normalize();
    Vec3f y = (z ^ x).normalize();

    Matrix res = Matrix::identity(4);
    for (int i = 0; i < 3; i++) {
        res[0][i] = x[i];
        res[1][i] = y[i];
        res[2][i] = z[i];
    }

    res[0][3] = -(x * m_position);
    res[1][3] = -(y * m_position);
    res[2][3] = -(z * m_position);
    return res;
}

Matrix Camera::projectionMatrix() const {
    Matrix proj = Matrix::identity();

    // «ум из FOV: чем меньше угол, тем сильнее "приближение"
    float fovRad = m_fovY * float(M_PI) / 180.f;
    float s = 1.f / std::tan(fovRad * 0.5f);  // классический "scale" из FOV

    proj[0][0] = s;
    proj[1][1] = s;

    // “от же самый tinyRenderer-хак дл€ перспективы по Z/W
    float dist = (m_position - m_target).norm();
    proj[3][2] = -1.f / dist;

    return proj;
}
