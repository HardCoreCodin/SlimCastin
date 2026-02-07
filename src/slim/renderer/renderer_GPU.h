#pragma once

#include "./pixel_shader.h"


#define USE_GPU_BY_DEFAULT true
#define SLIM_THREADS_PER_BLOCK 256

struct DeviceHits {
    WallHit* wall_hits;
    GroundHit* ground_hits;
};

__constant__ RayCasterSettings d_settings;
__constant__ u32* d_window_content;
__constant__ DeviceHits d_hits;
__constant__ Slice<Circle> d_columns;
__constant__ Slice<LocalEdge> d_local_edges;

RayCasterSettings t_settings;
Slice<Circle> t_columns;
Slice<LocalEdge> t_local_edges;
DeviceHits t_hits;
TextureMip *t_texture_mips;
TexelQuad *t_texel_quads;
u32* t_window_content;

__global__ void d_generateWallHits(RayCaster ray_caster) {
    const u32 x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= ray_caster.screen_width)
        return;

    WallHit wall_hit;
    RayHit closest_hit;
    Ray ray;
    vec2 ray_direction = ray_caster.first_ray_direction + (f32)x * ray_caster.right_step;
    ray_caster.generateWallHit(wall_hit, ray_direction, ray, closest_hit, d_local_edges, d_columns);
    d_hits.wall_hits[x] = wall_hit;
}

void generateWallHitsOnGPU(const RayCaster& ray_caster) {
    u32 pixel_count = ray_caster.screen_width;
    u32 threads = SLIM_THREADS_PER_BLOCK;
    u32 blocks  = pixel_count / threads;
    if (pixel_count < threads) {
        threads = pixel_count;
        blocks = 1;
    } else if (pixel_count % threads)
        blocks++;

    d_generateWallHits<<<blocks, threads>>>(ray_caster);
}

__global__ void d_render(const RayCaster ray_caster) {
    const u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (ray_caster.screen_width * ray_caster.screen_height)) return;

    const u16 x = (u16)(i % ray_caster.screen_width);
    const u16 y = (u16)(i / ray_caster.screen_width);

    const WallHit &wall_hit = d_hits.wall_hits[x];
    const GroundHit &ground_hit = d_hits.ground_hits[y];

    Color pixel;
    if (y < wall_hit.top ||
        y > wall_hit.bot)
        renderGroundPixel(ground_hit, ray_caster.position, wall_hit.ray_direction, y < ray_caster.mid_point, d_settings, pixel);
    else
        renderWallPixel(wall_hit, y, d_settings, pixel);

    d_window_content[ray_caster.screen_width * y  + x] = pixel.asContent();
}

void renderOnGPU(const RayCaster& ray_caster, u32* window_content) {
    u32 pixel_count = ray_caster.screen_width * ray_caster.screen_height;
    u32 threads = SLIM_THREADS_PER_BLOCK;
    u32 blocks  = pixel_count / threads;
    if (pixel_count < threads) {
        threads = pixel_count;
        blocks = 1;
    } else if (pixel_count % threads)
        blocks++;

    d_render<<<blocks, threads>>>(ray_caster);

    checkErrors()
    downloadN(t_window_content, window_content, pixel_count * 2)
}


void initDataOnGPU(const RayCasterSettings& settings) {
    t_settings = settings;
    gpuErrchk(cudaMalloc(&t_window_content, sizeof(u32) * MAX_WINDOW_SIZE * 4))
    uploadConstant(&t_window_content, d_window_content)

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
    gpuErrchk(cudaMalloc(&t_local_edges.data,    sizeof(LocalEdge)  * MAX_TILE_MAP_EDGES))
    gpuErrchk(cudaMalloc(&t_columns.data,    sizeof(Circle)  * MAX_COLUMN_COUNT))

    uploadConstant(&t_hits, d_hits);
    uploadConstant(&t_local_edges, d_local_edges);
    uploadConstant(&t_columns, d_columns);

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

void uploadSettings(const RayCasterSettings* settings) {
    t_settings.render_mode = settings->render_mode;
    uploadConstant(&t_settings, d_settings)
}

void uploadLocalEdges(const Slice<LocalEdge>& local_edges) {
    t_local_edges.size = local_edges.size;
    uploadConstant(&t_local_edges, d_local_edges);
    uploadN(local_edges.data, t_local_edges.data, local_edges.size)
}

void uploadColumns(const Slice<Circle>& columns) {
    t_columns.size = columns.size;
    uploadConstant(&t_columns, d_columns);
    uploadN(columns.data, t_columns.data, columns.size)
}

void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {
    uploadN(ground_hits, t_hits.ground_hits, ground_hits_count)
}

void uploadWallHits(WallHit* wall_hits, u16 wall_hits_count) {
    uploadN(wall_hits, t_hits.wall_hits, wall_hits_count)
}

void downloadWallHits(WallHit* wall_hits, u16 wall_hits_count) {
    downloadN(t_hits.wall_hits, wall_hits, wall_hits_count)
}
