#pragma once

#include "constants.h"
#include "../math/vec3.h"


// Fade function for smooth interpolation (like Perlin's fade)
INLINE_XPU float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

INLINE_XPU f32 grad(int hash, f32 x, f32 y, f32 z) {
    int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
    f32 u = h<8 ? x : y,                 // INTO 12 GRADIENT DIRECTIONS.
        v = h<4 ? y : h==12||h==14 ? x : z;
    return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}


INLINE_XPU f32 perlinNoise(const vec3& V, const u8* p) {
    f32 x = V.x;
    f32 y = V.y;
    f32 z = V.z;
    int X = (int)floor(x) & 255,                  // FIND UNIT CUBE THAT
        Y = (int)floor(y) & 255,                  // CONTAINS POINT.
        Z = (int)floor(z) & 255;
    x -= floor(x);                                // FIND RELATIVE X,Y,Z
    y -= floor(y);                                // OF POINT IN CUBE.
    z -= floor(z);
    f32 u = fade(x),                                // COMPUTE FADE CURVES
    v = fade(y),                                // FOR EACH OF X,Y,Z.
    w = fade(z);
    int A = p[X  ]+Y, AA = p[A]+Z, AB = p[A+1]+Z,      // HASH COORDINATES OF
        B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;      // THE 8 CUBE CORNERS,
    return lerp(
        lerp(
            lerp(
                grad(p[AA  ], x  , y  , z   ),  // AND ADD
                grad(p[BA  ], x-1, y  , z   ),
                u
            ), // BLENDED
            lerp(
                grad(p[AB  ], x  , y-1, z   ),  // RESULTS
                grad(p[BB  ], x-1, y-1, z   ),
                u
            ),
            v
        ),// FROM  8
        lerp(
            lerp(
                grad(p[AA+1], x  , y  , z-1 ),  // CORNERS
                grad(p[BA+1], x-1, y  , z-1 ),
                u
            ), // OF CUBE
            lerp(
                grad(p[AB+1], x  , y-1, z-1 ),
                grad(p[BB+1], x-1, y-1, z-1 ),
                u
            ),
            v
        ),
        w
    );
}


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


enum class Edit : u8 {
    None = 0,
    Walls,
    Columns,
    Enemies,
};


struct RenderState {
    PointLight lights[MAX_POINT_LIGHTS];
    PointLight enemies[MAX_ENEMIES];
    vec3 lights_through_portal_from[MAX_POINT_LIGHTS];
    vec3 lights_through_portal_to[MAX_POINT_LIGHTS];
    vec2 hovered_pos;
    f32 time, parallax_occlusion_scale, rounded_corners_radius, rounded_corners_scale;
    RenderMode render_mode;
    BRDFType brdf;
    Edit edit;
    u16 screen_width;
    u16 screen_height;
    u8 flags;
    u8 light_count;
    u8 enemy_count;
    u8 step_count;
    u8 parallax_occlusion_max_steps;
    u8 parallax_occlusion_min_steps;
    u8 p[512];

    void init() {
        edit = Edit::None;
        brdf = BRDF_GGX;
        flags = USE_MAPS_MASK;
#ifdef __CUDACC__
        flags |= CAST_SHADOWS | VOLUMETRIC;
        step_count = 128;
        parallax_occlusion_max_steps = 64;
        parallax_occlusion_min_steps = 32;
#else
        step_count = 1;
        parallax_occlusion_max_steps = 32;
        parallax_occlusion_min_steps = 8;
#endif
        render_mode = DEFAULT_RENDER_MODE;
        hovered_pos = 0.0f;
        light_count = 1;
        enemy_count = 0;
        time = 0.0f;
        parallax_occlusion_scale = 0.14f;
        rounded_corners_radius = 0.2f;

        u8 permutation[] = { 151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
            129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
            49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        };
        for (int i=0; i < 256 ; i++) p[256+i] = p[i] = permutation[i];
    }

    INLINE_XPU f32 noise(const vec3& P) const {
        return 0.5f + 0.5f * perlinNoise(P + time, p);
    }
};