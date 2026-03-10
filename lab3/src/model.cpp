#include "model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>
#include <algorithm>
#include <cmath>

// Получение директории из полного пути к файлу
static std::string GetBaseDir(const std::string& path) {
    const size_t s1 = path.find_last_of('/');
    const size_t s2 = path.find_last_of('\\');
    const size_t s = (s1 == std::string::npos) ? s2 : (s2 == std::string::npos ? s1 : std::max(s1, s2));
    if (s == std::string::npos) return std::string();
    return path.substr(0, s + 1);
}

Model::Model(const char* filename)
    : verts_()
    , norms_()
    , uvs_()
    , tris_()
    , diffusemap_() {

    if (!filename) return;

    tinyobj::ObjReaderConfig config;
    config.triangulate = true;           // Разбиваем полигоны на треугольники
    config.mtl_search_path = GetBaseDir(filename); // Путь для поиска материалов

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config)) {
        if (!reader.Error().empty())
            std::cerr << "TinyObj error: " << reader.Error() << "\n";
        return;
    }
    if (!reader.Warning().empty())
        std::cerr << "TinyObj warn: " << reader.Warning() << "\n";

    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

    // Загрузка вершин
    verts_.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3) {
        verts_.push_back(Vec3f(attrib.vertices[i + 0], attrib.vertices[i + 1], attrib.vertices[i + 2]));
    }

    // Загрузка текстурных координат
    uvs_.reserve(attrib.texcoords.size() / 2);
    for (size_t i = 0; i + 1 < attrib.texcoords.size(); i += 2) {
        uvs_.push_back(Vec2f(attrib.texcoords[i + 0], attrib.texcoords[i + 1]));
    }

    // Загрузка нормалей
    norms_.reserve(attrib.normals.size() / 3);
    for (size_t i = 0; i + 2 < attrib.normals.size(); i += 3) {
        norms_.push_back(Vec3f(attrib.normals[i + 0], attrib.normals[i + 1], attrib.normals[i + 2]));
    }

    // Функция для создания треугольника из индексов
    auto emitTri = [&](const tinyobj::index_t& i0,
        const tinyobj::index_t& i1,
        const tinyobj::index_t& i2) {
            Tri t{};
            t.v[0] = i0.vertex_index;
            t.v[1] = i1.vertex_index;
            t.v[2] = i2.vertex_index;

            t.vt[0] = i0.texcoord_index;
            t.vt[1] = i1.texcoord_index;
            t.vt[2] = i2.texcoord_index;

            t.vn[0] = i0.normal_index;
            t.vn[1] = i1.normal_index;
            t.vn[2] = i2.normal_index;

            // Генерация нормалей, если они отсутствуют
            bool needGen = norms_.empty() || (t.vn[0] < 0) || (t.vn[1] < 0) || (t.vn[2] < 0);

            if (needGen) {
                Vec3f faceN(0.f, 0.f, 1.f);

                if (t.v[0] >= 0 && t.v[1] >= 0 && t.v[2] >= 0 &&
                    t.v[0] < (int)verts_.size() && t.v[1] < (int)verts_.size() && t.v[2] < (int)verts_.size()) {

                    const Vec3f a = verts_[t.v[0]];
                    const Vec3f b = verts_[t.v[1]];
                    const Vec3f c = verts_[t.v[2]];
                    faceN = ((b - a) ^ (c - a));
                    if (faceN.norm() > 1e-12f) faceN.normalize();
                    else faceN = Vec3f(0.f, 0.f, 1.f);
                }

                const int newIdx = (int)norms_.size();
                norms_.push_back(faceN);

                if (t.vn[0] < 0) t.vn[0] = newIdx;
                if (t.vn[1] < 0) t.vn[1] = newIdx;
                if (t.vn[2] < 0) t.vn[2] = newIdx;
            }

            tris_.push_back(t);
        };

    // Обработка всех граней из всех shapes
    for (const tinyobj::shape_t& shape : shapes) {
        size_t index_offset = 0;
        const auto& num_face_vertices = shape.mesh.num_face_vertices;
        const auto& indices = shape.mesh.indices;

        for (size_t f = 0; f < num_face_vertices.size(); f++) {
            const int fv = num_face_vertices[f];
            if (fv < 3) { index_offset += (size_t)fv; continue; }

            // Сборка полигона из индексов
            std::vector<tinyobj::index_t> poly;
            poly.reserve((size_t)fv);
            for (int v = 0; v < fv; v++) poly.push_back(indices[index_offset + (size_t)v]);

            // Триангуляция полигона
            if (fv == 3) {
                emitTri(poly[0], poly[1], poly[2]);
            }
            else {
                for (int i = 1; i + 1 < fv; i++) emitTri(poly[0], poly[(size_t)i], poly[(size_t)i + 1]);
            }

            index_offset += (size_t)fv;
        }
    }

    // Вывод статистики загрузки
    std::cerr << "# v# " << verts_.size()
        << " tri# " << tris_.size()
        << " vt# " << uvs_.size()
        << " vn# " << norms_.size()
        << std::endl;

    load_texture(filename, "_diffuse.tga", diffusemap_);
}

Model::~Model() {}

int Model::nverts() { return (int)verts_.size(); }
int Model::nfaces() { return (int)tris_.size(); }

Vec3f Model::vert(int i) { return verts_[i]; }

// Получение индексов вершин грани
std::vector<int> Model::face(int idx) {
    const Tri& t = tris_[idx];
    return { t.v[0], t.v[1], t.v[2] };
}

// Получение нормали вершины
Vec3f Model::norm(int iface, int nvert) {
    const Tri& t = tris_[iface];
    int ni = t.vn[nvert];
    if (ni < 0 || ni >= (int)norms_.size()) return Vec3f(0.f, 0.f, 1.f);
    Vec3f n = norms_[ni];
    return n.normalize();
}

// Получение текстурных координат в пикселях
Vec2i Model::uv(int iface, int nvert) {
    const int w = diffusemap_.get_width();
    const int h = diffusemap_.get_height();
    if (w <= 0 || h <= 0) return Vec2i(0, 0);

    const Tri& t = tris_[iface];
    int ti = t.vt[nvert];
    if (ti < 0 || ti >= (int)uvs_.size()) return Vec2i(0, 0);

    float u = uvs_[ti].x;
    float v = uvs_[ti].y;

    // Кламп текстурных координат в диапазон [0,1]
    u = std::max(0.f, std::min(1.f, u));
    v = std::max(0.f, std::min(1.f, v));

    // Преобразование UV в координаты текстуры
    int x = (int)std::round(u * (float)(w - 1));
    int y = (int)std::round(v * (float)(h - 1));

    x = std::max(0, std::min(w - 1, x));
    y = std::max(0, std::min(h - 1, y));
    return Vec2i(x, y);
}

// Получение цвета из диффузной карты
TGAColor Model::diffuse(const Vec2i& uv) {
    const int w = diffusemap_.get_width();
    const int h = diffusemap_.get_height();
    if (w <= 0 || h <= 0) return TGAColor(200, 200, 200);

    int x = std::max(0, std::min(w - 1, uv.x));
    int y = std::max(0, std::min(h - 1, uv.y));
    return diffusemap_.get(x, y);
}

// Загрузка текстуры из файла
void Model::load_texture(const std::string& objFilename, const char* suffix, TGAImage& img) {
    std::string texfile(objFilename);
    size_t dot = texfile.find_last_of('.');
    if (dot != std::string::npos) texfile = texfile.substr(0, dot);
    texfile += suffix;

    std::cerr << "texture file " << texfile << " loading "
        << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;

    if (img.get_width() > 0 && img.get_height() > 0) img.flip_vertically();
}