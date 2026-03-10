#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include <string>

#include "geometry.h"
#include "tgaimage.h"

class Model {
private:
    struct Tri {
        int v[3];
        int vt[3];
        int vn[3];
    };

private:
    std::vector<Vec3f> verts_;
    std::vector<Vec3f> norms_;
    std::vector<Vec2f> uvs_;
    std::vector<Tri>   tris_;

    TGAImage diffusemap_;

    void load_texture(const std::string& objFilename, const char* suffix, TGAImage& img);

public:
    Model(const char* filename);
    ~Model();

    int nverts();
    int nfaces();

    Vec3f vert(int i);
    std::vector<int> face(int idx);

    Vec3f norm(int iface, int nvert);
    Vec2i uv(int iface, int nvert);

    TGAColor diffuse(const Vec2i& uv);
};

#endif 
