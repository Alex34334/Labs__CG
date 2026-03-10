#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

#include "tgaimage.h"
#include "geometry.h"
#include "camera.h"
#include "model.h"

static const int kWidth = 800;
static const int kHeight = 800;
static const int kDepth = 255;

static Matrix viewport(int x, int y, int w, int h) {
    Matrix m = Matrix::identity();
    m[0][0] = w / 2.f;      m[0][3] = x + w / 2.f;
    m[1][1] = h / 2.f;      m[1][3] = y + h / 2.f;
    m[2][2] = kDepth / 2.f; m[2][3] = kDepth / 2.f;
    return m;
}

static Vec3f barycentric(const Vec3f* pts, const Vec3f& P) {
    Vec3f u = (Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x) ^
        Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y));
    if (std::fabs(u.z) < 1e-2f) return Vec3f(-1.f, 1.f, 1.f);
    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
}

struct VtxIn {
    Vec3f view_pos;
    Vec3f world_pos;
    Vec3f normal;
    Vec2f uv_px;
};

static std::vector<VtxIn> clip_near_plane_viewspace(const std::vector<VtxIn>& poly, float zNear) {
    auto inside = [&](const VtxIn& v) { return v.view_pos.z <= -zNear; };

    std::vector<VtxIn> out;
    out.reserve(poly.size() + 2);

    if (poly.empty()) return out;

    VtxIn S = poly.back();
    bool S_in = inside(S);

    for (const VtxIn& E : poly) {
        bool E_in = inside(E);

        if (S_in && E_in) {
            out.push_back(E);
        }
        else if (S_in && !E_in) {
            float zS = S.view_pos.z;
            float zE = E.view_pos.z;
            float t = (-zNear - zS) / (zE - zS);

            VtxIn I;
            I.view_pos = S.view_pos + (E.view_pos - S.view_pos) * t;
            I.world_pos = S.world_pos + (E.world_pos - S.world_pos) * t;
            I.normal = S.normal + (E.normal - S.normal) * t;
            I.uv_px = S.uv_px + (E.uv_px - S.uv_px) * t;
            out.push_back(I);
        }
        else if (!S_in && E_in) {
            float zS = S.view_pos.z;
            float zE = E.view_pos.z;
            float t = (-zNear - zS) / (zE - zS);

            VtxIn I;
            I.view_pos = S.view_pos + (E.view_pos - S.view_pos) * t;
            I.world_pos = S.world_pos + (E.world_pos - S.world_pos) * t;
            I.normal = S.normal + (E.normal - S.normal) * t;
            I.uv_px = S.uv_px + (E.uv_px - S.uv_px) * t;
            out.push_back(I);
            out.push_back(E);
        }

        S = E;
        S_in = E_in;
    }

    return out;
}

struct RVtx {
    Vec3f screen;
    float invw = 0.f;
    Vec3f world_over_w;
    Vec3f norm_over_w;
    Vec2f uv_over_w;
};

static RVtx make_raster_vtx(const VtxIn& v, const Matrix& Proj, const Matrix& VP) {
    RVtx o;

    Vec4f clip = Proj * Vec4f(v.view_pos.x, v.view_pos.y, v.view_pos.z, 1.f);

    float w = clip.w;
    if (std::fabs(w) < 1e-8f) w = (w >= 0.f ? 1e-8f : -1e-8f);
    o.invw = 1.f / w;

    Vec4f ndc(clip.x * o.invw, clip.y * o.invw, clip.z * o.invw, 1.f);
    Vec4f scr = VP * ndc;
    o.screen = Vec3f(scr.x, scr.y, scr.z);

    o.world_over_w = v.world_pos * o.invw;
    o.norm_over_w = v.normal * o.invw;
    o.uv_over_w = v.uv_px * o.invw;

    return o;
}

static TGAColor shade_phong(const Model& model,
    const Vec3f& P_world,
    const Vec3f& N_world,
    const Vec2i& uv_px,
    const Vec3f& eye_world,
    const Vec3f& light_dir_world) {
    Vec3f N = N_world;
    N.normalize();

    Vec3f L = light_dir_world;
    L.normalize();

    Vec3f V = (eye_world - P_world).normalize();
    Vec3f R = (N * (2.f * (N * L)) - L).normalize();

    const float ambient = 0.20f;
    const float diff = std::max(0.f, N * L);
    const float spec = std::pow(std::max(0.f, R * V), 32.f);

    const float intensity = ambient + diff + 0.40f * spec;

    TGAColor base = const_cast<Model&>(model).diffuse(uv_px);
    return base * intensity;
}

// Растеризация треугольника с поддержкой режима записи
static void raster_triangle(const RVtx& a, const RVtx& b, const RVtx& c,
    const Model& model,
    TGAImage& image,
    std::vector<float>& zbuffer,
    const Vec3f& eye_world,
    const Vec3f& light_dir_world,
    std::vector<bool>& pixel_drawn,
    bool recordOnly) {
    Vec3f pts[3] = { a.screen, b.screen, c.screen };

    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    Vec2f clamp((float)kWidth - 1.f, (float)kHeight - 1.f);

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

            float z = a.screen.z * bc.x + b.screen.z * bc.y + c.screen.z * bc.z;

            int idx = x + y * kWidth;
            if (idx < 0 || idx >= kWidth * kHeight) continue;

            if (z < zbuffer[idx]) {
                if (recordOnly) {
                    // Режим сбора: только отмечаем пиксель
                    pixel_drawn[idx] = true;
                }
                else {
                    // Режим отрисовки: нормальная растеризация
                    float invw = a.invw * bc.x + b.invw * bc.y + c.invw * bc.z;
                    if (std::fabs(invw) < 1e-12f) continue;

                    Vec3f world = (a.world_over_w * bc.x + b.world_over_w * bc.y + c.world_over_w * bc.z) / invw;
                    Vec3f norm = (a.norm_over_w * bc.x + b.norm_over_w * bc.y + c.norm_over_w * bc.z) / invw;
                    Vec2f uvf = (a.uv_over_w * bc.x + b.uv_over_w * bc.y + c.uv_over_w * bc.z) / invw;

                    Vec2i uv_px((int)std::round(uvf.x), (int)std::round(uvf.y));

                    TGAColor col = shade_phong(model, world, norm, uv_px, eye_world, light_dir_world);

                    zbuffer[idx] = z;
                    image.set(x, y, col);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* obj_path = (argc >= 2) ? argv[1] : "./obj/african_head.obj";
    const char* out_path = (argc >= 3) ? argv[2] : "output.tga";

    Model model(obj_path);
    if (model.nverts() <= 0 || model.nfaces() <= 0) {
        std::cerr << "Failed to load OBJ or empty mesh: " << obj_path << "\n";
        return 1;
    }

    // Вычисление ограничивающего бокса модели
    Vec3f bbmin(std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    Vec3f bbmax(-std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max());

    for (int i = 0; i < model.nverts(); i++) {
        Vec3f v = model.vert(i);
        bbmin.x = std::min(bbmin.x, v.x); bbmin.y = std::min(bbmin.y, v.y); bbmin.z = std::min(bbmin.z, v.z);
        bbmax.x = std::max(bbmax.x, v.x); bbmax.y = std::max(bbmax.y, v.y); bbmax.z = std::max(bbmax.z, v.z);
    }

    Vec3f center((bbmin.x + bbmax.x) * 0.5f,
        (bbmin.y + bbmax.y) * 0.5f,
        (bbmin.z + bbmax.z) * 0.5f);

    Vec3f extent(bbmax.x - bbmin.x, bbmax.y - bbmin.y, bbmax.z - bbmin.z);
    float radius = std::max(extent.x, std::max(extent.y, extent.z)) * 0.5f;
    float scale = (radius > 1e-6f) ? (1.0f / radius) : 1.0f;

    Vec3f eye_world(0.0f, 0.0f, 3.0f);
    Vec3f target_world(0.0f, 0.0f, 0.0f);

    float fov = 75.f;
    float aspect = (float)kWidth / (float)kHeight;
    float zNear = 0.01f;
    float zFar = 50.f;

    Camera camera(eye_world, target_world, Vec3f(0.f, 1.f, 0.f), fov, aspect, zNear, zFar);

    Matrix View = camera.viewMatrix();
    Matrix Proj = camera.projectionMatrix();
    Matrix VP = viewport(0, 0, kWidth, kHeight);

    TGAImage image(kWidth, kHeight, TGAImage::RGB);
    std::vector<float> zbuffer((size_t)kWidth * (size_t)kHeight,
        std::numeric_limits<float>::infinity());

    Vec3f light_dir = Vec3f(0.2f, -1.f, 0.1f).normalize();

    // Массив для отслеживания закрашенных пикселей
    std::vector<bool> pixel_drawn(kWidth * kHeight, false);

    // --- ПЕРВЫЙ ПРОХОД: только сбор границ ---
    std::cout << "First pass: collecting bounds..." << std::endl;
    for (int iface = 0; iface < model.nfaces(); ++iface) {
        std::vector<int> f = model.face(iface);

        std::vector<VtxIn> poly;
        poly.reserve(3);

        for (int j = 0; j < 3; ++j) {
            Vec3f v_raw = model.vert(f[j]);
            Vec3f world = (v_raw - center) * scale;

            Vec4f view4 = View * Vec4f(world.x, world.y, world.z, 1.f);
            VtxIn vin;
            vin.view_pos = Vec3f(view4.x, view4.y, view4.z);
            vin.world_pos = world;
            vin.normal = model.norm(iface, j);
            Vec2i uv_i = model.uv(iface, j);
            vin.uv_px = Vec2f((float)uv_i.x, (float)uv_i.y);
            poly.push_back(vin);
        }

        std::vector<VtxIn> clipped = clip_near_plane_viewspace(poly, zNear);
        if (clipped.size() < 3) continue;

        RVtx v0 = make_raster_vtx(clipped[0], Proj, VP);
        for (size_t k = 1; k + 1 < clipped.size(); ++k) {
            RVtx v1 = make_raster_vtx(clipped[k], Proj, VP);
            RVtx v2 = make_raster_vtx(clipped[k + 1], Proj, VP);

            Vec3f A = v1.screen - v0.screen;
            Vec3f B = v2.screen - v0.screen;
            if (((A ^ B).norm()) < 1e-6f) continue;

            raster_triangle(v0, v1, v2, model, image, zbuffer, eye_world, light_dir, pixel_drawn, true);
        }
    }

    // --- ПОИСК ГРАНИЦ ЗАКРАШЕННОЙ ОБЛАСТИ ---
    int minX = kWidth, minY = kHeight, maxX = 0, maxY = 0;
    bool any_drawn = false;

    for (int y = 0; y < kHeight; y++) {
        for (int x = 0; x < kWidth; x++) {
            if (pixel_drawn[x + y * kWidth]) {
                any_drawn = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    // --- ЗАЛИВКА СИНЕГО ПРЯМОУГОЛЬНИКА ---
    if (any_drawn) {
        std::cout << "Filling blue rectangle from (" << minX << "," << minY << ") to (" << maxX << "," << maxY << ")" << std::endl;
        TGAColor blue(0, 0, 255, 255);
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                image.set(x, y, blue);
            }
        }
    }
    else {
        std::cout << "No pixels drawn?" << std::endl;
    }

    // --- ВТОРОЙ ПРОХОД: финальная отрисовка ---
    std::cout << "Second pass: rendering model..." << std::endl;
    // Сброс z-буфера для второго прохода
    std::fill(zbuffer.begin(), zbuffer.end(), std::numeric_limits<float>::infinity());

    for (int iface = 0; iface < model.nfaces(); ++iface) {
        std::vector<int> f = model.face(iface);

        std::vector<VtxIn> poly;
        poly.reserve(3);

        for (int j = 0; j < 3; ++j) {
            Vec3f v_raw = model.vert(f[j]);
            Vec3f world = (v_raw - center) * scale;

            Vec4f view4 = View * Vec4f(world.x, world.y, world.z, 1.f);
            VtxIn vin;
            vin.view_pos = Vec3f(view4.x, view4.y, view4.z);
            vin.world_pos = world;
            vin.normal = model.norm(iface, j);
            Vec2i uv_i = model.uv(iface, j);
            vin.uv_px = Vec2f((float)uv_i.x, (float)uv_i.y);
            poly.push_back(vin);
        }

        std::vector<VtxIn> clipped = clip_near_plane_viewspace(poly, zNear);
        if (clipped.size() < 3) continue;

        RVtx v0 = make_raster_vtx(clipped[0], Proj, VP);
        for (size_t k = 1; k + 1 < clipped.size(); ++k) {
            RVtx v1 = make_raster_vtx(clipped[k], Proj, VP);
            RVtx v2 = make_raster_vtx(clipped[k + 1], Proj, VP);

            Vec3f A = v1.screen - v0.screen;
            Vec3f B = v2.screen - v0.screen;
            if (((A ^ B).norm()) < 1e-6f) continue;

            raster_triangle(v0, v1, v2, model, image, zbuffer, eye_world, light_dir, pixel_drawn, false);
        }
    }

    image.flip_vertically();
    if (!image.write_tga_file(out_path)) {
        std::cerr << "Failed to write TGA: " << out_path << "\n";
        return 2;
    }

    std::cout << "Saved: " << out_path << "\n";
    return 0;
}