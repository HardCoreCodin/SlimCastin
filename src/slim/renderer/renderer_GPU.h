#pragma once

#include "./pixel_shader.h"


#define USE_GPU_BY_DEFAULT true
#define SLIM_THREADS_PER_BLOCK 64

struct DeviceHits {
    WallHit* wall_hits;
    GroundHit* ground_hits;
};

__constant__ RayCasterSettings d_settings;
__constant__ CanvasData d_canvas;
__constant__ DeviceHits d_hits;

RayCasterSettings t_settings;
CanvasData t_canvas;
DeviceHits t_hits;
TextureMip *t_texture_mips;
TexelQuad *t_texel_quads;

__global__ void d_render(vec2 position) {
    const u32 s = d_canvas.antialias == SSAA ? 2 : 1;
    const u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    const u32 half_screen_height = d_canvas.dimensions.height >> 1;
    const u32 screen_height = half_screen_height << 1;
    const u32 screen_width = d_canvas.dimensions.width;
    if (i >= ((screen_width * half_screen_height) * s * s))
    return;

    u16 x = (u16)(i % (screen_width * s));
    u16 y = (u16)(i / (screen_width * s));

    const vec2 tile_map_end = vec2((f32)(d_settings.tile_map_width - 1), (f32)(d_settings.tile_map_height - 1));
    renderPixel(x, y, position, tile_map_end,
        d_canvas.pixels, screen_width, screen_height,
        d_hits.wall_hits[x], d_hits.ground_hits[y],
        d_settings.textures,
        d_settings.ceiling_texture_id,
        d_settings.floor_texture_id);
}

void renderOnGPU(const Canvas &canvas, vec2 position) {
    t_canvas.dimensions = canvas.dimensions;
    t_canvas.antialias = canvas.antialias;
    uploadConstant(&t_canvas, d_canvas)

    u32 sample_count = canvas.dimensions.width * (canvas.dimensions.height >> 1);
    u32 pixel_count = sample_count * (canvas.antialias == SSAA ? 4 : 1);
    // u32 depths_count = sample_count * (canvas.antialias == NoAA ? 1 : 4);
    u32 threads = SLIM_THREADS_PER_BLOCK;
    u32 blocks  = pixel_count / threads;
    if (pixel_count < threads) {
        threads = pixel_count;
        blocks = 1;
    } else if (pixel_count % threads)
        blocks++;

    d_render<<<blocks, threads>>>(position);

    checkErrors()
    downloadN(t_canvas.pixels, canvas.pixels, pixel_count * 2)
    // downloadN(t_canvas.depths, canvas.depths, depths_count)
}


void initDataOnGPU(const RayCasterSettings& settings) {
    t_settings = settings;
    gpuErrchk(cudaMalloc(&t_canvas.pixels, sizeof(Pixel) * MAX_WINDOW_SIZE * 4))
    // gpuErrchk(cudaMalloc(&t_canvas.depths, sizeof(f32) * MAX_WINDOW_SIZE * 4))

    u32 total_mip_count = 0;
    u32 total_texel_quads_count = 0;
    Texture *texture = settings.textures;
    for (u32 i = 0; i < settings.textures_count; i++, texture++) {
        total_mip_count += texture->mip_count;
        TextureMip *mip = texture->mips;
        for (u32 m = 0; m < texture->mip_count; m++, mip++)
            total_texel_quads_count += (mip->width + 1) * (mip->height + 1);
    }
    gpuErrchk(cudaMalloc(&t_settings.textures,  sizeof(Texture) * settings.textures_count))
    gpuErrchk(cudaMalloc(&t_texture_mips,   sizeof(TextureMip) * total_mip_count))
    gpuErrchk(cudaMalloc(&t_texel_quads,    sizeof(TexelQuad)  * total_texel_quads_count))
    gpuErrchk(cudaMalloc(&t_hits.wall_hits,   sizeof(WallHit) * MAX_WALL_HITS_COUNT))
    gpuErrchk(cudaMalloc(&t_hits.ground_hits,    sizeof(GroundHit)  * MAX_GROUND_HITS_COUNT))

    uploadConstant(&t_hits, d_hits);

    TexelQuad *d_quads = t_texel_quads;
    TextureMip *d_mips = t_texture_mips;
    Texture *t_textures = t_settings.textures;
    Texture t_texture;
    texture = settings.textures;
    for (u32 i = 0; i < settings.textures_count; i++, texture++) {
        t_texture = *texture;
        t_texture.mips = d_mips;
        uploadN(&t_texture, t_textures, 1)
        t_textures++;

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

    uploadConstant(&t_settings, d_settings)
}

void uploadWallHits(WallHit* wall_hits, u16 wall_hits_count) {
    uploadN(wall_hits, t_hits.wall_hits, wall_hits_count)
}

void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {
    uploadN(ground_hits, t_hits.ground_hits, ground_hits_count)
}
