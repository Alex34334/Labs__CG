#include "camera.h"
#include <cmath>

constexpr float kPi = 3.14159265358979323846f;


//Построение look-at матрицы камеры

static Matrix lookat(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
    // Вычисление базисных векторов камеры
    Vec3f z = (eye - center).normalize();
    Vec3f x = (up ^ z).normalize();
    Vec3f y = (z ^ x).normalize();

    // Заполнение матрицы вида
    Matrix m = Matrix::identity();
    m[0][0] = x.x; m[0][1] = x.y; m[0][2] = x.z; m[0][3] = -(x * eye);
    m[1][0] = y.x; m[1][1] = y.y; m[1][2] = y.z; m[1][3] = -(y * eye);
    m[2][0] = z.x; m[2][1] = z.y; m[2][2] = z.z; m[2][3] = -(z * eye);
    return m;
}


//Получение матрицы вида
Matrix Camera::viewMatrix() const {
    return lookat(m_eye, m_center, m_up);
}


// Построение матрицы перспективной проекции
 
Matrix Camera::projectionMatrix() const {
    Matrix p;
    // Расчет параметров проекции
    const float fovRad = m_fovY * float(kPi) / 180.f;
    const float f = 1.f / std::tan(fovRad * 0.5f);

    // Заполнение матрицы проекции
    p[0][0] = f / m_aspect;
    p[1][1] = f;

    p[2][2] = (m_zFar + m_zNear) / (m_zNear - m_zFar);
    p[2][3] = (2.f * m_zFar * m_zNear) / (m_zNear - m_zFar);

    p[3][2] = -1.f;
    p[3][3] = 0.f;

    return p;
}