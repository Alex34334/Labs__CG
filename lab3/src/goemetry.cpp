#include "geometry.h"

// Конструктор - обнуляем все элементы матрицы
Matrix::Matrix() {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m[i][j] = 0.f;
}

// Создание единичной матрицы 4x4
Matrix Matrix::identity() {
    Matrix I;
    I[0][0] = 1.f;
    I[1][1] = 1.f;
    I[2][2] = 1.f;
    I[3][3] = 1.f;
    return I;
}

// Умножение матриц (стандартное матричное произведение)
Matrix Matrix::operator*(const Matrix& r) const {
    Matrix res;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++) s += m[i][k] * r.m[k][j];
            res.m[i][j] = s;
        }
    }
    return res;
}

// Умножение матрицы на 4D вектор
Vec4f operator*(const Matrix& a, const Vec4f& v) {
    Vec4f r;
    r.x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z + a.m[0][3] * v.w;
    r.y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z + a.m[1][3] * v.w;
    r.z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z + a.m[2][3] * v.w;
    r.w = a.m[3][0] * v.x + a.m[3][1] * v.y + a.m[3][2] * v.z + a.m[3][3] * v.w;
    return r;
}