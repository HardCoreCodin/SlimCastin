#pragma once

#include "./pixel_shader.h"
#include "./raycast.h"


#define USE_GPU_BY_DEFAULT true
#define SLIM_THREADS_PER_BLOCK 256

struct DeviceHits {
    WallHitGroup* wall_hits;
    GroundHit* ground_hits;
};

__constant__ RenderData d_render_data;
__constant__ u32* d_window_content;
__constant__ DeviceHits d_hits;
__constant__ Slice<Circle> d_columns;
__constant__ Slice<TileEdge> d_edges;

RenderData t_render_data;
Slice<Circle> t_columns;
Slice<TileEdge> t_edges;
DeviceHits t_hits;
TextureMip *t_texture_mips;
TexelQuad *t_texel_quads;
u32* t_window_content;

__global__ void d_generateWallHits(RayCast raycast) {
    const u32 x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= raycast.screen_width)
        return;

    vec2 ray_direction = raycast.first_ray_direction + (f32)x * raycast.right_step;
    raycast.generateWallHit(d_hits.wall_hits[x], ray_direction, d_edges, d_columns);
}

void generateWallHitsOnGPU(const RayCast& raycast) {
    u32 pixel_count = raycast.screen_width;
    u32 threads = SLIM_THREADS_PER_BLOCK;
    u32 blocks  = pixel_count / threads;
    if (pixel_count < threads) {
        threads = pixel_count;
        blocks = 1;
    } else if (pixel_count % threads)
        blocks++;

    d_generateWallHits<<<blocks, threads>>>(raycast);
}

__global__ void d_render(const RayCast raycast, const RenderState render_state) {
    const u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (raycast.screen_width * raycast.screen_height)) return;

    const u16 x = (u16)(i % raycast.screen_width);
    const u16 y = (u16)(i / raycast.screen_width);

    PixelShader pixel_shader{d_render_data, render_state};
    const WallHitGroup& wall_hit_group{d_hits.wall_hits[x]};
    const Pixel pixel = pixel_shader.shade(
        d_hits.ground_hits[y],
        wall_hit_group.main,
        raycast.portals,
        d_edges,
        d_columns,
        raycast.position,
        y,
        raycast.mid_point,
        wall_hit_group.portal,
        wall_hit_group.portal_origin);

    d_window_content[raycast.screen_width * y + x] = pixel.asContent();
}

void renderOnGPU(const RayCast& ray_caster, const RenderState& render_state, u32* window_content) {
    u32 pixel_count = ray_caster.screen_width * ray_caster.screen_height;
    u32 threads = SLIM_THREADS_PER_BLOCK;
    u32 blocks  = pixel_count / threads;
    if (pixel_count < threads) {
        threads = pixel_count;
        blocks = 1;
    } else if (pixel_count % threads)
        blocks++;

    d_render<<<blocks, threads>>>(ray_caster, render_state);

    checkErrors()
    downloadN(t_window_content, window_content, pixel_count * 2)
}


void initDataOnGPU(const RenderData& render_data) {
    t_render_data = render_data;
    gpuErrchk(cudaMalloc(&t_window_content, sizeof(u32) * MAX_WINDOW_SIZE * 4))
    uploadConstant(&t_window_content, d_window_content)

    u32 total_mip_count = 0;
    u32 total_texel_quads_count = 0;
    const Texture *texture = render_data.textures;
    for (u32 i = 0; i < render_data.texture_count; i++, texture++) {
        total_mip_count += texture->mip_count;
        TextureMip *mip = texture->mips;
        for (u32 m = 0; m < texture->mip_count; m++, mip++)
            total_texel_quads_count += (mip->width + 1) * (mip->height + 1);
    }
    gpuErrchk(cudaMalloc(&t_render_data.textures,  sizeof(Texture) * render_data.texture_count))
    gpuErrchk(cudaMalloc(&t_texture_mips,   sizeof(TextureMip) * total_mip_count))
    gpuErrchk(cudaMalloc(&t_texel_quads,    sizeof(TexelQuad)  * total_texel_quads_count))
    gpuErrchk(cudaMalloc(&t_hits.wall_hits,   sizeof(WallHitGroup) * MAX_WALL_HITS_COUNT))
    gpuErrchk(cudaMalloc(&t_hits.ground_hits,    sizeof(GroundHit)  * MAX_GROUND_HITS_COUNT))
    gpuErrchk(cudaMalloc(&t_edges.data,    sizeof(TileEdge)  * MAX_TILE_MAP_EDGES))
    gpuErrchk(cudaMalloc(&t_columns.data,    sizeof(Circle)  * MAX_COLUMN_COUNT))

    uploadConstant(&t_hits, d_hits);
    uploadConstant(&t_edges, d_edges);
    uploadConstant(&t_columns, d_columns);

    TexelQuad *d_quads = t_texel_quads;
    TextureMip *d_mips = t_texture_mips;
    Texture *t_textures = t_render_data.textures;
    Texture t_texture;
    texture = render_data.textures;
    for (u32 i = 0; i < render_data.texture_count; i++, texture++) {
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

    uploadConstant(&t_render_data, d_render_data)
}

void uploadEdges(const Slice<TileEdge>& edges) {
    t_edges.size = edges.size;
    uploadConstant(&t_edges, d_edges);
    uploadN(edges.data, t_edges.data, t_edges.size)
}

void uploadColumns(const Slice<Circle>& columns) {
    t_columns.size = columns.size;
    uploadConstant(&t_columns, d_columns);
    uploadN(columns.data, t_columns.data, columns.size)
}

void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {
    uploadN(ground_hits, t_hits.ground_hits, ground_hits_count)
}

void uploadWallHits(WallHitGroup* wall_hits, u16 wall_hits_count) {
    uploadN(wall_hits, t_hits.wall_hits, wall_hits_count)
}

void downloadWallHits(WallHitGroup* wall_hits, u16 wall_hits_count) {
    downloadN(t_hits.wall_hits, wall_hits, wall_hits_count)
}
