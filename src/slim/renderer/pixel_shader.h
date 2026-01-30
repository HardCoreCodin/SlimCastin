#pragma once

#include "./render_data.h"


INLINE_XPU void renderPixel(u16 x, u16 y, vec2 position, vec2 tile_map_end,
    Pixel* pixels, u16 screen_width, u16 screen_height,
    const WallHit& wall_hit, const GroundHit& ground_hit,
    const Texture* textures, u16 top_texture_id, u16 bot_texture_id) {

    u8 mip = 255;
    f32 u, v_top, v_bot;
    vec2 hit_position;
    if (y < wall_hit.top) {
        hit_position = position + wall_hit.ray_direction * ground_hit.z;
        if (inRange({}, hit_position, tile_map_end)) {
            u             = hit_position.x - (f32)(i32)hit_position.x;
            v_top = v_bot = hit_position.y - (f32)(i32)hit_position.y;
            mip = ground_hit.mip;
        }
    } else {
        u = wall_hit.u;
        v_top = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
        if (v_top > 1.0f) v_top = 1.0f;
        v_bot = 1.0f - v_top;
        mip = wall_hit.mip;
        hit_position = wall_hit.hit_position;
        top_texture_id = bot_texture_id = wall_hit.texture_id;
    }

    u32 offset_top = screen_width *                      y  + x;
    u32 offset_bot = screen_width * (screen_height - 1 - y) + x;
    if (mip == 255) {
        Pixel magenta{1.0f, 0.0f, 1.0f, 1.0f};
        pixels[offset_top] = magenta;
        pixels[offset_bot] = magenta;
    } else {
        f32 dim_factor = 0.25f + (hit_position - position).squaredLength();
        dim_factor = dim_factor < 1.0f ? 1.0f : dim_factor;
        dim_factor = 1.5f / dim_factor;
        pixels[offset_top] = textures[top_texture_id].mips[mip].sample(u, v_top) * dim_factor;
        pixels[offset_bot] = textures[bot_texture_id].mips[mip].sample(u, v_bot) * dim_factor;
    }
}