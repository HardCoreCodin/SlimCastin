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


struct RenderState {
    PointLight lights[MAX_POINT_LIGHTS];
    vec3 lights_through_portal_from[MAX_POINT_LIGHTS];
    vec3 lights_through_portal_to[MAX_POINT_LIGHTS];
    vec2 hovered_pos;
    RenderMode render_mode;
    u8 light_count;
    u8 flags;

    void init() {
        flags = (u8)BRDF_GGX | USE_MAPS_MASK;
#ifdef __CUDACC__
        flags |= CAST_SHADOWS;
#endif
        render_mode = DEFAULT_RENDER_MODE;
        hovered_pos = 0.0f;
        light_count = 1;
    }
};