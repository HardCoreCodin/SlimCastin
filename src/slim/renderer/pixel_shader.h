#pragma once

#include "../math/vec2.h"

#define MAX_WALL_HITS_COUNT (1024*5)
#define MAX_GROUND_HITS_COUNT 1024

#define RAY_CASTER_DEFAULT_SETTINGS_RENDER_MODE RenderMode_Beauty
#define RAY_CASTER_DEFAULT_SETTINGS_FILTER_MODE FilterMode_BiLinear

enum FilterMode {
    FilterMode_None,
    FilterMode_BiLinear,
    FilterMode_TriLinear
};

struct RayCasterSettings {
    Texture* textures;
    u8 textures_count;
    u8 floor_texture_id;
    u8 ceiling_texture_id;
    u16 tile_map_width;
    u16 tile_map_height;

    FilterMode filter_mode;
    RenderMode render_mode;
    ColorID mip_level_colors[9];

    void init(
        const Slice<Texture>& textures_slice,
        u8 floor_texture,
        u8 ceiling_texture,
        u16 init_tile_map_width,
        u16 init_tile_map_height,
        FilterMode init_filter_mode = RAY_CASTER_DEFAULT_SETTINGS_FILTER_MODE,
        RenderMode init_render_mode = RAY_CASTER_DEFAULT_SETTINGS_RENDER_MODE) {
        textures = textures_slice.data;
        textures_count = (u8)textures_slice.size;
        floor_texture_id = floor_texture;
        ceiling_texture_id = ceiling_texture;
        tile_map_width = init_tile_map_width;
        tile_map_height = init_tile_map_height;
        filter_mode = init_filter_mode;
        render_mode = init_render_mode;

        mip_level_colors[0] = BrightRed;
        mip_level_colors[1] = BrightYellow;
        mip_level_colors[2] = BrightGreen;
        mip_level_colors[3] = BrightMagenta;
        mip_level_colors[4] = BrightCyan;
        mip_level_colors[5] = BrightBlue;
        mip_level_colors[6] = BrightGrey;
        mip_level_colors[7] = Grey;
        mip_level_colors[8] = DarkGrey;
    }
};

INLINE u8 min(u8 a, u8 b) { return a < b ? a : b; }
INLINE u8 max(u8 a, u8 b) { return a > b ? a : b; }
INLINE u8 clamp(u8 v, u8 min_v, u8 max_v) { return min(max(v, min_v), max_v); }
INLINE u8 closestLog2(u32 v) {
    u8 r = 0;
    while (v) {r++; v >>= 1;}
    return r;
}
INLINE u8 computeMip(f32 pixel_coverage, f32 texel_size, u8 last_mip) {
    return pixel_coverage < texel_size ? 0 : min(last_mip, closestLog2((u32)(pixel_coverage / texel_size) - 1));
}
INLINE_XPU bool inRange(vec2 start, vec2 value, vec2 end) {
    return inRange(start.x, value.x, end.x) &&
           inRange(start.y, value.y, end.y);
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


    INLINE void update(u16 screen_height, f32 texel_size, f32 pixel_coverage_factor, f32 column_height_factor, u8 last_mip, vec2 new_ray_direction, const RayHit &ray_hit) {
        ray_direction = new_ray_direction;
        texture_id = ray_hit.texture_id;

        u = ray_hit.tile_fraction;
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
    }
};


INLINE_XPU void renderPixel(u16 x, u16 y, vec2 position, vec2 tile_map_end,
    Pixel* pixels, u16 screen_width, u16 screen_height,
    const WallHit& wall_hit, const GroundHit& ground_hit,
    const Texture* textures, u16 top_texture_id, u16 bot_texture_id) {

    u8 mip = 255;
    f32 z, u, v_top, v_bot;
    if (y < wall_hit.top) {
        vec2 pos = position + wall_hit.ray_direction * ground_hit.z;
        if (inRange({}, pos, tile_map_end)) {
            u             = pos.x - (f32)(i32)pos.x;
            v_top = v_bot = pos.y - (f32)(i32)pos.y;
            z = ground_hit.z;
            mip = ground_hit.mip;
        }
    } else {
        u = wall_hit.u;
        v_top = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
        if (v_top > 1.0f) v_top = 1.0f;
        v_bot = 1.0f - v_top;
        z = wall_hit.z;
        mip = wall_hit.mip;
        top_texture_id = bot_texture_id = wall_hit.texture_id;
    }

    u32 offset_top = screen_width *                      y  + x;
    u32 offset_bot = screen_width * (screen_height - 1 - y) + x;
    if (mip == 255) {
        Pixel magenta{1.0f, 0.0f, 1.0f, 1.0f};
        pixels[offset_top] = magenta;
        pixels[offset_bot] = magenta;
    } else {
        f32 dim_factor = 0.25f + z * z;
        dim_factor = dim_factor < 1.0f ? 1.0f : dim_factor;
        dim_factor = 1.5f / dim_factor;
        pixels[offset_top] = textures[top_texture_id].mips[mip].sample(u, v_top) * dim_factor;
        pixels[offset_bot] = textures[bot_texture_id].mips[mip].sample(u, v_bot) * dim_factor;
    }
}