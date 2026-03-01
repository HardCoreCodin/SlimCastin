#pragma once

#include "./render_data.h"
#include "../math/vec3.h"


INLINE_XPU f32 ggxTrowbridgeReitz_D(f32 roughness, f32 NdotH) { // NDF
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    f32 a = roughness * roughness;
    f32 denom = NdotH * NdotH * (a - 1.0f) + 1.0f;
    return (
        a
        /
        (pi * denom * denom)
    );
}

INLINE_XPU f32 ggxSchlickSmith_G(f32 roughness, f32 NdotL, f32 NdotV, bool IBL = false) {
    // https://learnopengl.com/PBR/Theory
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    f32 a = roughness * roughness;
    f32 k = a * 0.5f; // Approximation from Karis (UE4)
    //    if (IBL) {
    //        k *= k * 0.5f;
    //    } else { // direct
    //        k += 1.0f;
    //        k *= k * 0.125f;
    //    }
    f32 one_minus_k = 1.0f - k;
    f32 denom = fast_mul_add(NdotV, one_minus_k, k);
    f32 result = NdotV / fmaxf(denom, EPS);
    denom = fast_mul_add(NdotL, one_minus_k, k);
    result *= NdotL / fmaxf(denom, EPS);
    return result;
}

INLINE_XPU f32 schlickFresnel(f32 HdotL, const f32 &F0) {
    return F0 + (1.0f - F0) * powf(1.0f - HdotL, 5.0f);
}


INLINE_XPU f32 GGX(f32 roughness, f32 NdotL, f32 NdotV, f32 NdotH) {
    float a2 = roughness * roughness;
    const f32 D = ggxTrowbridgeReitz_D(a2, NdotH);
    float G_V = NdotV + sqrt( (NdotV - NdotV * a2) * NdotV + a2 );
    float G_L = NdotL + sqrt( (NdotL - NdotL * a2) * NdotL + a2 );
    return D / ( G_V * G_L );
    //
    // const f32 G = ggxSchlickSmith_G(a2, NdotL, NdotV);
    // return D * G
    //           /
    //           (4.0f * NdotL * NdotV);
}

struct PixelShader {
    const RayCasterSettings& settings;
    Color pixel;

    INLINE_XPU const Color& shade(const GroundHit& ground_hit, const WallHit& wall_hit, const vec2& position, u16 y, i32 mid_point) {
        pixel = Magenta;

        if (!wall_hit.isValid()) return pixel;

        vec3 P, N, Ro;
        f32 u, v;
        u8 mip_level, texture_id, is;

        if (y < wall_hit.top ||
            y > wall_hit.bot) {
            const bool is_ceiling = y < mid_point;

            const vec2 ray_hit_position = position + wall_hit.ray_direction * ground_hit.z;
            const vec2 start = 1.0f;
            const vec2 end = {
                (f32)(settings.tile_map_width - 1),
                (f32)(settings.tile_map_height - 1)
            };
            if (!inRange(start, ray_hit_position, end)) return pixel;

            mip_level = ground_hit.mip;
            v = ray_hit_position.y - (f32)(i32)ray_hit_position.y;
            u = ray_hit_position.x - (f32)(i32)ray_hit_position.x;
            is = is_ceiling ? ABOVE : BELOW;
            texture_id = is_ceiling ? settings.ceiling_texture_id : settings.floor_texture_id;
            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;
            P.x = ray_hit_position.x;
            P.z = ray_hit_position.y;
            P.y = is_ceiling ? 1.0f : -1.0f;
        } else {
            mip_level = wall_hit.mip;
            v = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
            u = wall_hit.u;
            is = wall_hit.is;
            texture_id = wall_hit.texture_id;
            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;
            P.x = wall_hit.hit_position.x;
            P.z = wall_hit.hit_position.y;
            P.y = (1.0f - v) * 2.0f - 1.0f;
        }

        if (settings.render_mode == RenderMode_Beauty ||
            settings.render_mode == RenderMode_Color)
            pixel = settings.textures[texture_id].mips[mip_level].sampleColor(u, v);
        else if (settings.render_mode == RenderMode_Light)
            pixel = White;

        f32 roughness = 1.0f;
        if (settings.flags & USE_ROUGHNESS_MAP &&
            (settings.render_mode == RenderMode_Roughness ||
             settings.render_mode == RenderMode_Beauty ||
             settings.render_mode == RenderMode_Light)) {
            roughness = settings.textures[texture_id + 1].mips[mip_level].sampleColor(u, v).r;
        }

        N = {0.0f, 0.0f, 1.0f};
        if (settings.render_mode == RenderMode_Beauty ||
            settings.render_mode == RenderMode_Normal ||
            settings.render_mode == RenderMode_Light) {
            if (settings.flags & USE_NORMAL_MAP)
                N = vec3{settings.textures[texture_id + 2].mips[mip_level].sampleColor(u, v)}.scaleAdd(2.0f, -1.0f).normalized();
            if      (is & FACING_DOWN ) N = {   N.x,   N.y,    N.z};
            else if (is & FACING_UP   ) N = {  -N.x,   N.y,   -N.z};
            else if (is & FACING_LEFT ) N = {-N.z,   N.y,  N.x};
            else if (is & FACING_RIGHT) N = { N.z,   N.y, -N.x};
            else if (is & ABOVE)        N = {   N.x,-N.z, -N.y};
            else if (is & BELOW)        N = {   N.x, N.z, -N.y};
        }

        float AO = 1.0f;
        if (settings.flags & USE_AO_MAP &&
            (settings.render_mode == RenderMode_AO ||
             settings.render_mode == RenderMode_Beauty ||
             settings.render_mode == RenderMode_Light)) {
            AO = settings.textures[texture_id + 3].mips[mip_level].sampleColor(u, v).r;
            AO *= AO;
            AO *= AO;
        }

        switch (settings.render_mode) {
            case RenderMode_Color: break;
            case RenderMode_AO: pixel = AO; break;
            case RenderMode_UVs: pixel = Color(u, v, 0); break;
            case RenderMode_Depth: pixel = 1.0f / (Ro - P).length(); break;
            case RenderMode_Normal: pixel = N.scaleAdd(0.5, 0.5f).asColor(); break;
            case RenderMode_MipLevel: pixel = Color(settings.mip_level_colors[mip_level]); break;
            case RenderMode_Roughness: pixel = roughness; break;
            case RenderMode_Untextured: pixel = Color(
                is == ABOVE ?
                    settings.untextured_ceiling_color :
                    (is == BELOW ?
                        settings.untextured_floor_color :
                        settings.untextured_wall_color)); break;
            default: {
                const vec3 V{(Ro - P).normalized()};
                const vec3 LP = vec3{settings.light_position_x, settings.light_position_y, settings.light_position_z};
                vec3 L = LP - P;
                const f32 attenuation = 1.0f / L.squaredLength();
                L *= sqrt(attenuation);
                f32 Li = settings.light_intensity * attenuation * attenuation;

                Color light{settings.light_color_r, settings.light_color_g, settings.light_color_b};
                const f32 NdotL = clampedValue(N.dot(L));

                f32 Fs = 0.0f;
                f32 F = 0.0f;

                const BRDFType brdf{(BRDFType)(settings.flags & 3)};
                if (brdf == BRDF_GGX) {
                    const vec3 H = (L + V).normalized();
                    const f32 NdotH = clampedValue(N.dot(H));
                    F = schlickFresnel(clampedValue(H.dot(L)), 0.04f);
                    const f32 NdotV = clampedValue(N.dot(V));

                    if (NdotV > 0.0f && roughness > 0.0f) {
                        Fs = GGX(roughness, NdotL, NdotV, NdotH);
                    }
                } else if (brdf != BRDF_Lambert) {
                    F = roughness;
                    f32 exponent = 16.0f;
                    f32 specular_factor = 0.0f;
                    if (brdf == BRDF_Phong) {
                        exponent = 4.0f;
                        const vec3 R = (-V).reflectedAround(N);
                        specular_factor = clampedValue(R.dot(L));
                    } else { // BLINN
                        clampedValue(N.dot((L + V).normalized()));
                    }
                    if (specular_factor > 0.0f)
                        Fs = powf(specular_factor, exponent);
                }

                pixel *= Li * (0.1f * AO + light * (NdotL * lerp(Fs, ONE_OVER_PI, F)));
            }
        }

        return pixel;
    }
};