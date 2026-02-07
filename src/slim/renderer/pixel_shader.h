#pragma once

#include "./render_data.h"


INLINE_XPU void renderWallPixel(const WallHit& wall_hit, u16 y, const RayCasterSettings& settings, Color& pixel) {
    f32 v;
    if (settings.render_mode == RenderMode_Beauty ||
        settings.render_mode == RenderMode_UVs) {
        v = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
        if (v > 1.0f)
            v = 1.0f;
        if (v < 0.0f)
            v = 0.0f;
    }
    switch (settings.render_mode) {
        case RenderMode_Beauty: pixel = settings.textures[wall_hit.texture_id].mips[wall_hit.mip].sampleColor(wall_hit.u, v) * wall_hit.dim_factor; break;
        case RenderMode_UVs: pixel = Color(wall_hit.u, v, 0) * wall_hit.dim_factor; break;
        case RenderMode_Untextured: pixel = Color(settings.untextured_wall_color) * wall_hit.dim_factor; break;
        case RenderMode_MipLevel: pixel = Color(settings.mip_level_colors[wall_hit.mip]) * wall_hit.dim_factor; break;
        case RenderMode_Depth: pixel = wall_hit.dim_factor; break;
    }
}

INLINE_XPU void renderGroundPixel(const GroundHit& ground_hit, vec2 position, vec2 ray_direction, const bool is_ceiling, const RayCasterSettings& settings, Color& pixel) {
    vec2 uv;
    pixel = ground_hit.dim_factor;
    if (settings.render_mode == RenderMode_Beauty ||
        settings.render_mode == RenderMode_UVs) {
        position += ray_direction * ground_hit.z;
        if (!inRange(vec2{0.0f, 0.0f}, position, vec2{(f32)(settings.tile_map_width - 1), (f32)(settings.tile_map_height - 1)})) {
            pixel.green = 0.0f;
            pixel.blue = 1.0f;
            pixel.red = 1.0f;
            return;
        }
        uv.x = position.x - (f32)(i32)position.x;
        uv.y = position.y - (f32)(i32)position.y;
    }

    switch (settings.render_mode) {
        case RenderMode_Beauty: pixel = settings.textures[
            is_ceiling ? settings.ceiling_texture_id : settings.floor_texture_id
            ].mips[ground_hit.mip].sampleColor(uv.x, uv.y) * ground_hit.dim_factor; break;
        case RenderMode_Untextured: pixel = Color(
            is_ceiling ? settings.untextured_ceiling_color : settings.untextured_floor_color
            ) * ground_hit.dim_factor; break;
        case RenderMode_UVs: pixel = Color(uv.u, uv.v, 0) * ground_hit.dim_factor; break;
        case RenderMode_MipLevel: pixel = Color(settings.mip_level_colors[ground_hit.mip]) * ground_hit.dim_factor; break;
        case RenderMode_Depth: pixel = ground_hit.dim_factor; break;
    }
}


//
// INLINE_XPU void renderPixel(u16 x, u16 y, vec2 position, vec2 tile_map_end,
//     Pixel& top_pixel, Pixel& bot_pixel,
//     const WallHit& wall_hit, const GroundHit& ground_hit,
//     const Texture* textures, u16 top_texture_id, u16 bot_texture_id) {
//     u8 mip = 255;
//     f32 dim_factor, u, v_top, v_bot;
//     vec2 hit_position;
//     if (y < wall_hit.top) {
//         hit_position = position + wall_hit.ray_direction * ground_hit.z;
//         if (inRange({}, hit_position, tile_map_end)) {
//             u             = hit_position.x - (f32)(i32)hit_position.x;
//             v_top = v_bot = hit_position.y - (f32)(i32)hit_position.y;
//             mip = ground_hit.mip;
//             dim_factor = ground_hit.dim_factor;
//         }
//     } else {
//         u = wall_hit.u;
//         v_top = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
//         if (v_top > 1.0f) v_top = 1.0f;
//         v_bot = 1.0f - v_top;
//         mip = wall_hit.mip;
//         dim_factor = wall_hit.dim_factor;
//         // hit_position = wall_hit.hit_position;
//         top_texture_id = bot_texture_id = wall_hit.texture_id;
//     }
//
//     if (mip == 255) {
//         Pixel magenta{1.0f, 0.0f, 1.0f, 1.0f};
//         top_pixel = magenta;
//         top_pixel = bot_pixel = magenta;
//     } else {
//         top_pixel = textures[top_texture_id].mips[mip].sample(u, v_top) * dim_factor;
//         bot_pixel = textures[bot_texture_id].mips[mip].sample(u, v_bot) * dim_factor;
//         // top_pixel = pixel * dim_factor;
//         // bot_pixel = pixel * dim_factor;
//     }
// }