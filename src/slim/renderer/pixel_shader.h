#pragma once

#include "../math/vec2.h"


u8 min(u8 a, u8 b) { return a < b ? a : b; }
u8 max(u8 a, u8 b) { return a > b ? a : b; }
u8 clamp(u8 v, u8 min_v, u8 max_v) { return min(max(v, min_v), max_v); }
u8 closestLog2(u32 v) {
    u8 r = 0;
    while (v) {r++; v >>= 1;}
    return r;
}
u8 computeMip(f32 pixel_coverage, f32 texel_size, u8 last_mip) {
    return pixel_coverage < texel_size ? 0 : min(last_mip, closestLog2((u32)(pixel_coverage / texel_size) - 1));
}


struct GroundHit {
    f32 z;
    u8 mip;
    u8 flags;
};

struct WallHit {
    vec2 ray_direction;
    f32 u, v, z, texel_step;
    u16 top, height;
    u8 texture_id;
    u8 mip;


    void update(u16 screen_height, f32 texel_size, f32 pixel_coverage_factor, f32 column_height_factor, u8 last_mip, vec2 new_ray_direction, const RayHit &ray_hit) {
        ray_direction = new_ray_direction;
        texture_id = ray_hit.texture_id;

        u = ray_hit.tile_fraction;// * 2.0f;
        z = ray_hit.perp_distance;
        height = (u16)(column_height_factor / z);
        mip = computeMip(z * pixel_coverage_factor, texel_size, last_mip);
        texel_step = 1.0f / (f32)height;
        if (height < screen_height) {
            top = (screen_height - height) >> 1;
            v = 0.0f;
        } else {
            v = (f32)((height - screen_height) >> 1) / (f32)height;
            height = screen_height;
            top    = 0;
        }
        z *= 0.75f;
    }
};

INLINE_XPU void renderPixel(u16 x, u16 y, vec2 position,
    Pixel* pixels, u16 screen_width, u16 screen_height, u16 tile_map_width, u16 tile_map_height,
    const WallHit* wall_hits, const GroundHit* ground_hits,
    const Texture* wall_textures, const Texture* floor_texture, const Texture* ceiling_texture) {

    u32 offset = screen_width * y + x;
    vec2 pos;
    f32 u, v, dim_factor;
    WallHit wall_hit = wall_hits[x];
    if (y < wall_hit.top || y > (screen_height - wall_hit.top)) {
        GroundHit ground_hit = ground_hits[y < wall_hit.top ? y : (screen_height - y)];
        pos = position + wall_hit.ray_direction * ground_hit.z;
        if (inRange(0.0f, pos.x, (f32)(tile_map_width - 1)) &&
            inRange(0.0f, pos.y, (f32)(tile_map_height - 1))) {

            u = pos.x - (f32)(i32)pos.x;
            v = pos.y - (f32)(i32)pos.y;

            dim_factor = 0.25f + ground_hit.z * ground_hit.z;
            dim_factor = dim_factor < 1.0f ? 1.0f : dim_factor;
            dim_factor = 1.5f / dim_factor;

            if (y < wall_hit.top)
                pixels[offset] = ceiling_texture->mips[ground_hit.mip].sample(u, v) * dim_factor;
            else
                pixels[offset] = floor_texture->mips[ground_hit.mip].sample(u, v) * dim_factor;
        }
    } else {
        u = wall_hit.u;
        v = wall_hit.v + (f32)(y - wall_hit.top) * wall_hit.texel_step;
        dim_factor = 0.25f + wall_hit.z * wall_hit.z;
        dim_factor = dim_factor < 1.0f ? 1.0f : dim_factor;
        dim_factor = 1.5f / dim_factor;
        pixels[offset] = wall_textures[wall_hit.texture_id].mips[wall_hit.mip].sample(u, v) * dim_factor;
    }
}