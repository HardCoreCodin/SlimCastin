#pragma once

#include "constants.h"
#include "../math/vec3.h"


struct RenderData {
    Texture* textures;
    u16 texture_count;
    u16 map_width;
    u16 map_height;
    u8 floor_texture_id;
    u8 ceiling_texture_id;
};


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

struct Enemy : PointLight {
    vec2 sincos;

    INLINE_XPU Color enemyColor(const vec3& dir, const vec3 pos, const f32 time) const {
        vec2 L{dir.x, dir.z};
        f32 dist = L.length();
        L /= dist;
        const f32 angle = atan2(L.y, L.x);
        // const f32 x_wave = sin(angle * 35.0f + time * 3.0f + sin(dist * 45.0f + time * 2.0f));
        // const f32 y_wave = sin(pos.z * dist * 0.3f + time * (sincos.y + 1.0f) * 0.2f);
        // // dist = sin(dist * 33.0f + 13.0f*(1.0f + (time * 2.0f + x_wave * 0.4f + y_wave * 0.3f)));
        // dist = sin((dist + sin(time * 2.0f)) * 10.0f + x_wave);
        const f32 dist_x_wave = sin(dist * (3.0f + 3.5f * ((sin(pos.x * 4.0f) + 1.0f) + time * 2.0f))) + 1.0f;
        const f32 dist_y_wave = cos(dist * (2.0f + 2.0f * ((cos(pos.z * 5.0f) + 1.0f) + time * 1.5f))) + 1.0f;
        const f32 angle_wave = sin((angle + dist)  * 5.5f + time * 3.0f) + 1.0f;
        Color enemy_color = color * (9.0f + 7.0f * (
                0.75f * (sin(dist * 2.0f + time * 3.0f) + 1.0f) *
                angle_wave +
                0.3f * dist_x_wave *
                dist_y_wave
            )
        );//(dist * 0.0f + x_wave * 1.0f + y_wave * 0.0f));
        // enemy_color.r -= abs(dir.x * sincos.x);
        // enemy_color.g -= abs(dir.y * sincos.y);
        // enemy_color.b -= abs(dir.z * (sincos.x + sincos.y)) * 0.5f;
        return enemy_color;
    }
};

struct RenderState {
    PointLight lights[MAX_POINT_LIGHTS];
    Enemy enemies[MAX_POINT_LIGHTS];
    vec3 lights_through_portal_from[MAX_POINT_LIGHTS];
    vec3 lights_through_portal_to[MAX_POINT_LIGHTS];
    vec2 hovered_pos;
    f32 time;
    RenderMode render_mode;
    u16 screen_width;
    u16 screen_height;
    u8 light_count;
    u8 enemy_count;
    u8 flags;

    void init() {
        flags = (u8)BRDF_GGX | USE_MAPS_MASK;
#ifdef __CUDACC__
        flags |= CAST_SHADOWS;
#endif
        render_mode = DEFAULT_RENDER_MODE;
        hovered_pos = 0.0f;
        light_count = 1;
        enemy_count = 1;
    }
};