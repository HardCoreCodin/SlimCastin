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

#define EDITING_WALLS     (1 << 3)
#define EDITING_COLUMNS   (1 << 4)
#define USE_ROUGHNESS_MAP (1 << 5)
#define USE_AO_MAP        (1 << 6)
#define USE_NORMAL_MAP    (1 << 7)

#define USE_MAPS_MASK (USE_ROUGHNESS_MAP | USE_AO_MAP | USE_NORMAL_MAP)

#define MAX_POINT_LIGHTS 8


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
    f32 body_radius;
    f32 initial_column_radius;
    f32 projectile_speed;
    f32 projectile_radius;

    // FilterMode filter_mode;
    ColorID untextured_wall_color;
    ColorID untextured_floor_color;
    ColorID untextured_ceiling_color;
    ColorID mip_level_colors[9];

    void init(
        const Slice<Texture>& textures_slice,
        u8 floor_texture,
        u8 ceiling_texture,
        u16 init_tile_map_width,
        u16 init_tile_map_height) {
        // FilterMode init_filter_mode = RAY_CASTER_DEFAULT_SETTINGS_FILTER_MODE) {
        textures = textures_slice.data;
        textures_count = (u8)textures_slice.size;
        floor_texture_id = floor_texture;
        ceiling_texture_id = ceiling_texture;
        tile_map_width = init_tile_map_width;
        tile_map_height = init_tile_map_height;
        // filter_mode = init_filter_mode;

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

        body_radius = 0.2f;
        initial_column_radius = 0.1f;

        projectile_speed = 3.0f;
        projectile_radius = 0.2f;
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
    return u + 1.0f;
}


struct PointLight {
    vec3 position;
    Color color;
    f32 intensity;

    void flicker(const Color& light_color, const f32 light_intensity, const f32 time) {
        color = light_color;
        color.g -= sinf(time*29.0f) * 0.07f + cosf(time*29.0f) * 0.07f;
        color.b -= sinf(time*19.0f) * 0.06f + cosf(time*19.0f) * 0.06f;
        intensity = light_intensity * 0.95f + sinf(time*17.0f) * light_intensity * 0.095f + cosf(time*23.0f) * light_intensity * 0.125f;
    }
};


struct RenderState {
    PointLight lights[8];
    RenderMode render_mode;
    u8 light_count;
    u8 flags;
    vec2 hovered_pos;

    void init() {
        flags = (u8)BRDF_GGX | USE_MAPS_MASK;
        render_mode = RAY_CASTER_DEFAULT_SETTINGS_RENDER_MODE;
        hovered_pos = 0.0f;
        light_count = 1;
    }
};


struct SpinningProjectile {
    vec3 position, forward, right;
    f32 angle, spawned_time;

    void init(const vec2 tile_map_position, const vec2 tile_map_forward, const f32 up_aim, const f32 projectile_radius, const f32 time) {
        position.x = tile_map_position.x;
        position.z = tile_map_position.y;
        position.y = 0.0f;

        forward.x = tile_map_forward.x;
        forward.z = tile_map_forward.y;
        forward.y = up_aim;
        forward = forward.normalized();

        spawned_time = time;
    }

    void updatePosition(const f32 travel) {
        position += forward * travel;
    }
};


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
            position = ray_direction * distance;
            perp_distance = forward.dot(position);
            position += ray_origin;
            tile_coords.x = (i32)position.x;
            tile_coords.y = (i32)position.y;
            texture_u = getU(position - columns[column_id].position);
            texture_u *= columns[column_id].radius;
            texture_id = 12;
            edge_is = 0;
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
    vec2 ray_direction, hit_position, hit_normal;
    f32 u, v, texel_step;
    u16 top, bot;
    u8 texture_id;
    u8 mip;
    u8 is;
    u8 column_id;

    INLINE_XPU void init() {
        v = -1.0f;
    }

    INLINE_XPU bool isValid() const {
        return v >= 0.0f;
    }

    INLINE_XPU void update(u16 screen_height, f32 texel_size, f32 pixel_coverage_factor, f32 column_height_factor, u8 last_mip, vec2 new_ray_direction, i32 mid_point, const Circle* columns, const RayHit &ray_hit) {
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

        if (bot >= screen_height)
            bot = screen_height - 1;

        is = ray_hit.edge_is;
        hit_position = ray_hit.position;
        column_id = ray_hit.column_id;
        if (column_id != INVALID_COLUMN_ID) {
            hit_normal = (hit_position - columns[column_id].position).normalized();
        }
    }
};