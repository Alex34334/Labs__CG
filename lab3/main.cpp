#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "camera.h"

const int width = 800;
const int height = 800;
const int depth = 255;

Model* model = nullptr;
float* zbuffer = nullptr;

Vec3f light_dir = Vec3f(1.f, -1.f, 1.f).normalize();

Matrix viewport(int x, int y, int w, int h) {
    Matrix m = Matrix::identity();
    m[0][0] = w / 2.f;
    m[0][3] = x + w / 2.f;
    m[1][1] = h / 2.f;
    m[1][3] = y + h / 2.f;
    m[2][2] = depth / 2.f;
    m[2][3] = depth / 2.f;
    return m;
}

Vec3f barycentric(const Vec3f* pts, const Vec3f& P) {
    Vec3f u = (Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x) ^
        Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y));
    if (std::fabs(u.z) < 1e-2f) return Vec3f(-1.f, 1.f, 1.f);
    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
}

struct IShader {
    virtual ~IShader() {}
    virtual Vec3f vertex(int iface, int nthvert) = 0;
    virtual bool fragment(const Vec3f& bar, TGAColor& color) = 0;
};

struct PhongShader : public IShader {
    Model& model;

    Matrix uniform_M;
    Matrix uniform_P;
    Matrix uniform_VP;

    Vec3f  uniform_light_dir;
    Vec3f  uniform_eye;

    Vec3f  uniform_center;
    float  uniform_scale;

    Vec3f varying_world_pos[3];
    Vec3f varying_normal[3];
    Vec2f varying_uv[3];

    PhongShader(Model& m,
        const Matrix& modelView,
        const Matrix& projection,
        const Matrix& viewport,
        const Vec3f& light_dir,
        const Vec3f& eye,
        const Vec3f& center,
        float scale)
        : model(m)
        , uniform_M(modelView)
        , uniform_P(projection)
        , uniform_VP(viewport)
        , uniform_light_dir(light_dir)
        , uniform_eye(eye)
        , uniform_center(center)
        , uniform_scale(scale) {
        uniform_light_dir.normalize();
    }

    Vec3f vertex(int iface, int nthvert) override {
        const std::vector<int> face = model.face(iface);
        int v_idx = face[nthvert];

        Vec3f v_raw = model.vert(v_idx);
        Vec3f v = (v_raw - uniform_center) * uniform_scale;
        varying_world_pos[nthvert] = v;

        Vec3f n = model.norm(iface, nthvert).normalize();
        varying_normal[nthvert] = n;

        Vec2i uv_i = model.uv(iface, nthvert);
        varying_uv[nthvert] = Vec2f((float)uv_i.x, (float)uv_i.y);

        Vec4f v4(v.x, v.y, v.z, 1.f);
        Vec4f view = uniform_M * v4;
        Vec4f clip = uniform_P * view;

        float w = (std::fabs(clip.w) > 1e-6f) ? clip.w : 1.f;
        clip.x /= w;
        clip.y /= w;
        clip.z /= w;

        Vec4f screen = uniform_VP * clip;

        return Vec3f(screen.x, screen.y, screen.z);
    }

    bool fragment(const Vec3f& bar, TGAColor& color) override {
        Vec3f p = varying_world_pos[0] * bar.x +
            varying_world_pos[1] * bar.y +
            varying_world_pos[2] * bar.z;

        Vec3f n = (varying_normal[0] * bar.x +
            varying_normal[1] * bar.y +
            varying_normal[2] * bar.z).normalize();

        TGAColor base(200, 200, 200);

        Vec3f L = uniform_light_dir;
        Vec3f V = (uniform_eye - p).normalize();
        Vec3f R = (n * (2.f * (n * L)) - L).normalize();

        float ambient = 0.2f;
        float diff = std::max(0.f, n * L);
        float spec = std::pow(std::max(0.f, R * V), 32.f);

        float intensity = ambient + diff + 0.4f * spec;
        color = base * intensity;
        return false;
    }
};

void triangle(Vec3f* pts, IShader& shader, TGAImage& image, float* zbuffer) {
    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    Vec2f clamp(width - 1.f, height - 1.f);

    for (int i = 0; i < 3; i++) {
        bboxmin.x = std::max(0.f, std::min(bboxmin.x, pts[i].x));
        bboxmin.y = std::max(0.f, std::min(bboxmin.y, pts[i].y));
        bboxmax.x = std::min(clamp.x, std::max(bboxmax.x, pts[i].x));
        bboxmax.y = std::min(clamp.y, std::max(bboxmax.y, pts[i].y));
    }

    for (int x = (int)bboxmin.x; x <= (int)bboxmax.x; x++) {
        for (int y = (int)bboxmin.y; y <= (int)bboxmax.y; y++) {
            Vec3f P((float)x + 0.5f, (float)y + 0.5f, 0.f);
            Vec3f bc = barycentric(pts, P);
            if (bc.x < 0.f || bc.y < 0.f || bc.z < 0.f) continue;

            float z = pts[0].z * bc.x +
                pts[1].z * bc.y +
                pts[2].z * bc.z;

            int idx = x + y * width;
            if (idx < 0 || idx >= width * height) continue;

            if (zbuffer[idx] < z) {
                TGAColor color;
                if (!shader.fragment(bc, color)) {
                    zbuffer[idx] = z;
                    image.set(x, y, color);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc == 2) model = new Model(argv[1]);
    else          model = new Model("obj/sponza.obj");

    zbuffer = new float[width * height];
    for (int i = 0; i < width * height; i++)
        zbuffer[i] = -std::numeric_limits<float>::infinity();

    Vec3f bbmin(std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    Vec3f bbmax(-std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max());
    for (int i = 0; i < model->nverts(); i++) {
        Vec3f v = model->vert(i);
        bbmin.x = std::min(bbmin.x, v.x);
        bbmin.y = std::min(bbmin.y, v.y);
        bbmin.z = std::min(bbmin.z, v.z);
        bbmax.x = std::max(bbmax.x, v.x);
        bbmax.y = std::max(bbmax.y, v.y);
        bbmax.z = std::max(bbmax.z, v.z);
    }
    Vec3f center = Vec3f((bbmin.x + bbmax.x) * 0.5f,
        (bbmin.y + bbmax.y) * 0.5f,
        (bbmin.z + bbmax.z) * 0.5f);
    Vec3f extent = Vec3f(bbmax.x - bbmin.x,
        bbmax.y - bbmin.y,
        bbmax.z - bbmin.z);
    float radius = std::max(extent.x, std::max(extent.y, extent.z)) * 0.5f;
    float scale = (radius > 1e-6f) ? (1.0f / radius) : 1.0f;

    Camera camera(
        Vec3f(0.f, 0.f, 5.f),
        Vec3f(0.f, 0.f, 0.f),
        Vec3f(0.f, 1.f, 0.f),
        60.f,
        (float)width / (float)height,
        0.1f,
        100.f
    );


    Matrix ModelView = camera.viewMatrix();
    Matrix Projection = camera.projectionMatrix();
    Matrix ViewPort = viewport(0, 0, width, height);

    TGAImage image(width, height, TGAImage::RGB);
    PhongShader shader(*model, ModelView, Projection, ViewPort, light_dir, camera.position(), center, scale);

    for (int i = 0; i < model->nfaces(); i++) {
        Vec3f screen_coords[3];
        for (int j = 0; j < 3; j++)
            screen_coords[j] = shader.vertex(i, j);
        triangle(screen_coords, shader, image, zbuffer);
    }

    image.flip_vertically();
    image.write_tga_file("output.tga");

    delete model;
    delete[] zbuffer;
    return 0;
}
