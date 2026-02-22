#pragma once

#include "../scene/tilemap_base.h"
#include "../math/vec3.h"

#define INVALID_EDGE_ID ((u16)(-1))
#define INVALID_COLUMN_ID ((u8)(-1))

#define MAX_WALL_HITS_COUNT (1024*5)
#define MAX_GROUND_HITS_COUNT (1024*2)

#define RAY_CASTER_DEFAULT_SETTINGS_RENDER_MODE RenderMode_Beauty
#define RAY_CASTER_DEFAULT_SETTINGS_FILTER_MODE FilterMode_BiLinear

INLINE_XPU bool inRange(i32 start, i32 value, i32 end) { return value >= start && value <= end; }
INLINE_XPU bool inRange(f32 start, f32 value, f32 end) { return value >= start && value <= end; }

#define USE_ROUGHNESS_MAP (1 << 5)
#define USE_AO_MAP        (1 << 6)
#define USE_NORMAL_MAP    (1 << 7)

#define USE_MAPS_MASK (USE_ROUGHNESS_MAP | USE_AO_MAP | USE_NORMAL_MAP)

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
    u8 flags;
    u16 tile_map_width;
    u16 tile_map_height;
    f32 light_intensity;
    f32 light_position_x;
    f32 light_position_y;
    f32 light_position_z;
    f32 light_color_r;
    f32 light_color_g;
    f32 light_color_b;

    FilterMode filter_mode;
    RenderMode render_mode;
    ColorID untextured_wall_color;
    ColorID untextured_floor_color;
    ColorID untextured_ceiling_color;
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

        untextured_wall_color = DarkGrey;
        untextured_floor_color = DarkYellow;
        untextured_ceiling_color = DarkCyan;

        flags = (u8)BRDF_CookTorrance | USE_MAPS_MASK;

        light_intensity = 4.0f;
    }
};

INLINE_XPU u8 min(u8 a, u8 b) { return a < b ? a : b; }
INLINE_XPU u8 max(u8 a, u8 b) { return a > b ? a : b; }
INLINE_XPU u8 clamp(u8 v, u8 min_v, u8 max_v) { return min(max(v, min_v), max_v); }
INLINE_XPU u8 closestLog2(u32 v) {
    u8 r = 0;
    while (v) {r++; v >>= 1;}
    return r;
}
INLINE_XPU u8 computeMip(f32 pixel_coverage, f32 texel_size, u8 last_mip) {
    return pixel_coverage < texel_size ? 0 : min(last_mip, closestLog2((u32)(pixel_coverage / texel_size) - 1));
}
INLINE_XPU bool inRange(vec2 start, vec2 value, vec2 end) {
    return inRange(start.x, value.x, end.x) &&
           inRange(start.y, value.y, end.y);
}

INLINE_XPU f32 getU(vec2 v) {
    f32 u = v.y / v.x;
    if (u > 1.0f || u < -1.0f) u = -1.0f / u;
    return (u + 1.0f) * 0.5f;
}

struct RayHit {
    vec2i tile_coords;
    vec2 position;

    f32 distance;
    f32 perp_distance;
    f32 texture_u;

    u16 local_edge_id;
    u8 column_id;
    u8 texture_id;
    u8 edge_is;

    INLINE_XPU void init() {
        column_id = INVALID_COLUMN_ID;
        local_edge_id = INVALID_EDGE_ID;
    }

    INLINE_XPU bool isValid() {
        return column_id != INVALID_COLUMN_ID ||
               local_edge_id != INVALID_EDGE_ID;
    }

    INLINE_XPU void finalize(const vec2 ray_origin, const vec2 ray_direction, const vec2 forward, const LocalEdge *local_edges, const Circle* columns) {
        if (column_id != INVALID_COLUMN_ID) {
            position = ray_origin + ray_direction * distance;
            tile_coords.x = (i32)position.x;
            tile_coords.y = (i32)position.y;
            texture_u = getU(position - columns[column_id].position);
            texture_u *= columns[column_id].radius;
            texture_id = 0;
            edge_is = 0;
            perp_distance = 0;
            return;
        }

        edge_is = local_edges[local_edge_id].is;
        texture_id = local_edges[local_edge_id].texture_id;

        vec2 local_hit_position = position;
        position += ray_origin;

        tile_coords.x = edge_is & FACING_RIGHT ? (i32)position.x - 1 : (i32)position.x;
        tile_coords.y = edge_is & FACING_DOWN  ? (i32)position.y - 1 : (i32)position.y;

        texture_u = edge_is & (FACING_LEFT | FACING_RIGHT) ?
            local_hit_position.y - local_edges[local_edge_id].from.y :
            local_hit_position.x - local_edges[local_edge_id].from.x;
        texture_u -= (f32)(i32)texture_u;
        if (edge_is & (FACING_RIGHT | FACING_UP))
            texture_u = 1.0f - texture_u;

        perp_distance = forward.dot(local_hit_position);
    }
};

struct GroundHit {
    f32 z;
    u8 mip;
    u8 flags;
};

struct WallHit {
    vec2 ray_direction, hit_position;
    f32 z2, u, v, texel_step;
    u16 top, bot;
    u8 texture_id;
    u8 mip;
    u8 is;

    INLINE_XPU void init() {
        z2 = -1.0f;
    }

    INLINE_XPU bool isValid() const {
        return z2 > 0.0f;
    }

    INLINE_XPU void update(u16 screen_height, f32 texel_size, f32 pixel_coverage_factor, f32 column_height_factor, u8 last_mip, vec2 new_ray_direction, i32 mid_point, const RayHit &ray_hit) {
        ray_direction = new_ray_direction;
        texture_id = ray_hit.texture_id;

        u = ray_hit.texture_u;
        v = 0.0f;

        u16 height = (u16)(column_height_factor / ray_hit.perp_distance);
        u16 half_height = height >> 1;
        height = half_height << 1;
        mip = computeMip(ray_hit.perp_distance * pixel_coverage_factor, texel_size, last_mip);
        texel_step = 1.0f / (f32)height;
        bot = (u16)mid_point + half_height;

        if (mid_point < half_height) {
            v = (f32)(half_height - mid_point) / (f32)height;
            top    = 0;
        }
        else top = (u16)mid_point - half_height;

        if (bot >= screen_height) {
            bot = screen_height - 1;
        }
        z2 = ray_direction.squaredLength() + ray_hit.perp_distance * ray_hit.perp_distance;
        is = ray_hit.edge_is;
        hit_position = ray_hit.position;
    }
};