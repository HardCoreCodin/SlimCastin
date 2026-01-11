#pragma once

#include "./ray_caster.h"


#define USE_GPU_BY_DEFAULT true
#define MESH_BVH_STACK_SIZE 16
#define SCENE_BVH_STACK_SIZE 6
#define SLIM_THREADS_PER_BLOCK 64

__constant__ SceneData d_scene;
__constant__ CanvasData d_canvas;

SceneData t_scene;
CanvasData t_canvas;
TextureMip *d_texture_mips;
TexelQuad *d_texel_quads;

__global__ void d_render(const RayCasterSettings settings, const CameraRayProjection projection) {
    u32 s = d_canvas.antialias == SSAA ? 2 : 1;
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (d_canvas.dimensions.width_times_height * s * s))
        return;

    Ray ray;
    RayHit hit;
    Color color;
    f32 depth;

    vec2i pixel_coords;
    pixel_coords.x = (i32)(i % ((u32)d_canvas.dimensions.width * s));
    pixel_coords.y = (i32)(i / ((u32)d_canvas.dimensions.width * s));

    hit.scaling_factor = 1.0f / sqrtf(projection.squared_distance_to_projection_plane +
        vec2{pixel_coords.x,
            -pixel_coords.y}.scaleAdd(projection.sample_size,projection.C_start).squaredLength());

    renderPixel(settings, projection, surface, ray, hit,
                projection.getRayDirectionAt(pixel_coords.x, pixel_coords.y), color, depth);

    ((Canvas*)(&d_canvas))->setPixel(pixel_coords.x, pixel_coords.y, color, -1, depth);
}

void renderOnGPU(const Canvas &canvas, const CameraRayProjection &projection, const RayCasterSettings &settings) {
    t_canvas.dimensions = canvas.dimensions;
    t_canvas.antialias = canvas.antialias;
    uploadConstant(&t_canvas, d_canvas)

    u32 pixel_count = canvas.dimensions.width_times_height * (canvas.antialias == SSAA ? 4 : 1);
    u32 depths_count = canvas.dimensions.width_times_height * (canvas.antialias == NoAA ? 1 : 4);
    u32 threads = SLIM_THREADS_PER_BLOCK;
    u32 blocks  = pixel_count / threads;
    if (pixel_count < threads) {
        threads = pixel_count;
        blocks = 1;
    } else if (pixel_count % threads)
        blocks++;

    d_render<<<blocks, threads>>>(settings, projection);

    checkErrors()
    downloadN(t_canvas.pixels, canvas.pixels, pixel_count)
    downloadN(t_canvas.depths, canvas.depths, depths_count)
}


void initDataOnGPU(const Scene &scene) {
    t_scene = scene;
    gpuErrchk(cudaMalloc(&t_canvas.pixels, sizeof(Pixel) * MAX_WINDOW_SIZE * 4))
    gpuErrchk(cudaMalloc(&t_canvas.depths, sizeof(f32) * MAX_WINDOW_SIZE * 4))
    // gpuErrchk(cudaMalloc(&t_scene.bvh_leaf_geometry_indices, sizeof(u32) * scene.counts.geometries))
    // gpuErrchk(cudaMalloc(&t_scene.bvh.nodes,sizeof(BVHNode)  * scene.counts.geometries * 2))


    if (scene.counts.textures) {
        u32 total_mip_count = 0;
        u32 total_texel_quads_count = 0;
        Texture *texture = scene.textures;
        for (u32 i = 0; i < scene.counts.textures; i++, texture++) {
            total_mip_count += texture->mip_count;
            TextureMip *mip = texture->mips;
            for (u32 m = 0; m < texture->mip_count; m++, mip++)
                total_texel_quads_count += (mip->width + 1) * (mip->height + 1);
        }
        gpuErrchk(cudaMalloc(&t_scene.textures, sizeof(Texture)    * scene.counts.textures))
        gpuErrchk(cudaMalloc(&d_texture_mips,   sizeof(TextureMip) * total_mip_count))
        gpuErrchk(cudaMalloc(&d_texel_quads,    sizeof(TexelQuad)  * total_texel_quads_count))

        TexelQuad *d_quads = d_texel_quads;
        TextureMip *d_mips = d_texture_mips;
        Texture *d_textures = t_scene.textures;
        Texture d_texture;
        texture = scene.textures;
        for (u32 i = 0; i < scene.counts.textures; i++, texture++) {
            d_texture = *texture;
            d_texture.mips = d_mips;
            uploadN(&d_texture, d_textures, 1)
            d_textures++;

            for (u32 m = 0; m < texture->mip_count; m++) {
                TextureMip mip = texture->mips[m];
                u32 quad_count = (mip.width + 1) * (mip.height + 1);
                uploadN( mip.texel_quads, d_quads, quad_count)

                mip.texel_quads = d_quads;
                uploadN(&mip, d_mips, 1)
                d_quads += quad_count;
                d_mips++;
            }
        }
    }

    uploadConstant(&t_scene, d_scene)
}
