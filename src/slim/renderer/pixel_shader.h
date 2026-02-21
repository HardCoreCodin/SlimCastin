#pragma once

#include "./render_data.h"
#include "../math/vec3.h"
#include "../math/vec4.h"


INLINE_XPU f32 light(f32 squared_distance, f32 light_intensity) {
    squared_distance *= squared_distance;
    squared_distance *= squared_distance;
    return light_intensity / squared_distance;
}
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

INLINE_XPU Color schlickFresnel(f32 HdotL, const Color &F0) {
    return F0 + (1.0f - F0) * powf(1.0f - HdotL, 5.0f);
}

INLINE_XPU Color cookTorrance(f32 roughness, f32 NdotL, f32 NdotV, f32 HdotL, f32 NdotH, const Color &F0, Color &F) {
    F = schlickFresnel(HdotL, F0);
    f32 D = ggxTrowbridgeReitz_D(roughness, NdotH);
    f32 G = ggxSchlickSmith_G(roughness, NdotL, NdotV);
    Color Ks = F * (D * G
              /
              (4.0f * NdotL * NdotV)
    );
    return Ks;
}
INLINE_XPU vec3 decodeNormal(const Color &color                                           ) {
    vec3 N = vec3{color.r, color.b, color.g};
    return (N * 2.0f -1.0f).normalized();
}
INLINE_XPU vec3 Cross(const vec3& lhs, const vec3& rhs) {
    return vec3{
        (lhs.y * rhs.z) - (lhs.z * rhs.y),
        (lhs.z * rhs.x) - (lhs.x * rhs.z),
        (lhs.x * rhs.y) - (lhs.y * rhs.x)
    };
}
INLINE_XPU vec3 rotateNormal(const vec3 &Ng, const Color &normal_sample) {
    vec3 Nm = decodeNormal(normal_sample);
    vec3 axis = vec3{Nm.z, 0, -Nm.x}.normalized();
    float angle = acosf(Nm.y);
    angle *= 0.5f;
    axis *= sinf(angle);
    float amount = cosf(angle);

    vec4 q;
    q.x = axis.x;
    q.y = axis.y;
    q.z = axis.z;
    q.w = amount;
    q = q.normalized();
    axis.x = q.x;
    axis.y = q.y;
    axis.z = q.z;
    amount = q.w;

    vec3 result{Cross(axis, Ng)};
    vec3 qqv{Cross(axis, result)};
    result = result * amount + qqv;
    result = result * 2.0f +  Ng;

    return result;
}

INLINE_XPU void shade(vec3 N, vec3 V, vec3 L, f32 Li, f32 roughness, Color& pixel) {
    // R = RF = ray.direction.reflectedAround(N);

    // const f32 NdotV = clampedValue(N.dot(V));
    const f32 NdotL = clampedValue(N.dot(L));
    const vec3 R = (-V).reflectedAround(N);
    // surface.F = schlickFresnel(clampedValue(surface.N.dot(surface.R)), surface.material->reflectivity);
    // Color F = schlickFresnel(clampedValue(N.dot(R)), 0.04f);

    Color Fs = Black;
    Color Fd = pixel;
    // if (material->brdf == BRDF_CookTorrance) {
        // Fd *= ONE_OVER_PI;// (1.0f - material->metalness) * ONE_OVER_PI;
        //
        // if (NdotV > 0.0f) { // TODO: This should not be necessary to check for, because rays would miss in that scenario so the code should never even get to this point - and yet it seems like it does.
        //     // If the viewing direction is perpendicular to the normal, no light can reflect
        //     // Both the numerator and denominator would evaluate to 0 in this case, resulting in NaN
        //
        //     // If roughness is 0 then the NDF (and hence the entire brdf) evaluates to 0
        //     // Otherwise, a negative roughness makes no sense logically and would be a user-error
        //     if (roughness > 0.0f) {
        //         const vec3 H = (L + V).normalized();
        //         const f32 NdotH = clampedValue(N.dot(H));
        //         const f32 HdotL = clampedValue(H.dot(L));
        //         // Fs = cookTorrance(material->roughness, NdotL, NdotV, HdotL, NdotH, material->reflectivity, F);
        //         Fs = cookTorrance(max(roughness - 0.2f, 0.0f), NdotL, NdotV, HdotL, NdotH, 0.04f, F);
        //         Fd *= 1.0f - F;
        //     }
        // }
    // }
    // else {
    Fd *= roughness * ONE_OVER_PI;

    // if (material->brdf != BRDF_Lambert) {
        f32 specular_factor, exponent;
        // if (material->brdf == BRDF_Phong) {
            exponent = 4.0f;
            specular_factor = clampedValue(R.dot(L));
        // } else {
        //     exponent = 16.0f;
        //     specular_factor = clampedValue(N.dot((L + V).normalized()));;
        // }
        if (specular_factor > 0.0f)
            Fs = 0.04f * (powf(specular_factor, exponent) * (1.0f - roughness));
    // }
    // }

    pixel = (Fs + Fd) * (NdotL * Li * Color(0.95f, 0.85f, 0.75f));
}


INLINE_XPU void renderWallPixel(const WallHit& wall_hit, u16 y, const RayCasterSettings& settings, Color& pixel) {
    const f32 v = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
    const f32 Pz = (0.5f - v) * 2.0f;
    const f32 z2 = wall_hit.z2 + Pz*Pz;
    // f32 v;
    // if (settings.render_mode == RenderMode_Beauty ||
    //     settings.render_mode == RenderMode_UVs ||
    //     settings.render_mode == RenderMode_Depth) {
    //     v = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
    //     if (v > 1.0f)
    //         v = 1.0f;
    //     if (v < 0.0f)
    //         v = 0.0f;
    //     }
    // f32 z2, Pz;
    // if (settings.render_mode == RenderMode_Beauty ||
    //     settings.render_mode == RenderMode_Depth) {
    //     Pz = 0.5 - v;
    //     Pz *= 2.0f;
    //     z2 = wall_hit.z2 + Pz*Pz;
    //     }
    switch (settings.render_mode) {
        case RenderMode_Beauty: {
            const f32 Li = light(z2, settings.light_intensity);
            const vec3 LP = vec3{settings.light_position_x, settings.light_position_z, settings.light_position_y};
            const vec3 P = vec3{wall_hit.hit_position.x, Pz, wall_hit.hit_position.y};
            const vec3 L = (LP - P).normalized();
            vec3 V = -vec3{wall_hit.ray_direction.x, Pz, wall_hit.ray_direction.y}.normalized();
            vec3 N = 0.0f;
            if (     wall_hit.is & FACING_UP)   N.z =  -1.0f;
            else if (wall_hit.is & FACING_DOWN) N.z = 1.0f;
            else if (wall_hit.is & FACING_LEFT) N.x = -1.0f;
            else if (wall_hit.is & FACING_RIGHT) N.x =  1.0f;
            pixel = settings.textures[wall_hit.texture_id].mips[wall_hit.mip].sampleColor(wall_hit.u, v);
            Color roughness = settings.textures[wall_hit.texture_id+1].mips[wall_hit.mip].sampleColor(wall_hit.u, v);
            Color normalMap = settings.textures[wall_hit.texture_id+2].mips[wall_hit.mip].sampleColor(wall_hit.u, v);
            Color AO = settings.textures[wall_hit.texture_id+3].mips[wall_hit.mip].sampleColor(wall_hit.u, v);
            N = rotateNormal(N, normalMap);
            shade(N, V, L, Li*AO.r, roughness.r, pixel);
            break;
        }
        case RenderMode_UVs: pixel = Color(wall_hit.u, v, 0); break;
        case RenderMode_Untextured: pixel = Color(settings.untextured_wall_color); break;
        case RenderMode_MipLevel: pixel = Color(settings.mip_level_colors[wall_hit.mip]); break;
        case RenderMode_Depth: pixel = 1.0f / sqrtf(z2); break;
    }
}

INLINE_XPU void renderGroundPixel(const GroundHit& ground_hit, vec2 position, vec2 ray_direction, const bool is_ceiling, const RayCasterSettings& settings, Color& pixel) {
    ray_direction *= ground_hit.z;
    position += ray_direction;
    if (!inRange(vec2{1.0f, 1.0f}, position, vec2{(f32)(settings.tile_map_width - 1), (f32)(settings.tile_map_height - 1)})) {
        return;
    }

    const f32 z2 = ray_direction.squaredLength() + 1.0f;
    const vec2 uv{
        position.x - (f32)(i32)position.x,
        position.y - (f32)(i32)position.y
    };
    // vec2 uv;
    // const f32 Pz = is_ceiling ? 1.0f : -1.0f;
    // const vec3 V = -vec3{ray_direction.x, ray_direction.y, Pz}.normalized();
    // if (settings.render_mode == RenderMode_Beauty ||
    //     settings.render_mode == RenderMode_UVs) {
    //     ray_direction *= ground_hit.z;
    //     position += ray_direction;
    //     if (!inRange(vec2{0.0f, 0.0f}, position, vec2{(f32)(settings.tile_map_width - 1), (f32)(settings.tile_map_height - 1)})) {
    //         pixel.green = 0.0f;
    //         pixel.blue = 1.0f;
    //         pixel.red = 1.0f;
    //         return;
    //     }
    //     uv.x = position.x - (f32)(i32)position.x;
    //     uv.y = position.y - (f32)(i32)position.y;
    //     }
    // f32 z2;
    // if (settings.render_mode == RenderMode_Beauty ||
    //     settings.render_mode == RenderMode_Depth) {
    //     z2 = ray_direction.squaredLength() + 1.0f;
    //     }
    switch (settings.render_mode) {
        case RenderMode_Beauty: {
            u8 texture_id = is_ceiling ? settings.ceiling_texture_id : settings.floor_texture_id;
            const vec3 V = -vec3{ray_direction.x, ray_direction.y, is_ceiling ? 1.0f : -1.0f}.normalized();
            const f32 Li = light(z2, settings.light_intensity);
            const vec3 LP = vec3{settings.light_position_x, settings.light_position_z, settings.light_position_y};
            const vec3 P = vec3{position.x, V.z, position.y};
            const vec3 L = (LP - P).normalized();
            vec3 N = vec3{0.0f, -V.z, 0.0f};
                      pixel = settings.textures[texture_id + 0].mips[ground_hit.mip].sampleColor(uv.x, uv.y);
            Color roughness = settings.textures[texture_id + 1].mips[ground_hit.mip].sampleColor(uv.x, uv.y);
            Color normalMap = settings.textures[texture_id + 2].mips[ground_hit.mip].sampleColor(uv.x, uv.y);
            Color AO        = settings.textures[texture_id + 3].mips[ground_hit.mip].sampleColor(uv.x, uv.y);
            N = rotateNormal(N, normalMap);
            shade(N, V, L, Li*AO.r, roughness.r, pixel);
            break;
        }
        case RenderMode_Untextured: pixel = Color(is_ceiling ? settings.untextured_ceiling_color : settings.untextured_floor_color); break;
        case RenderMode_UVs: pixel = Color(uv.u, uv.v, 0); break;
        case RenderMode_MipLevel: pixel = Color(settings.mip_level_colors[ground_hit.mip]); break;
        case RenderMode_Depth: pixel = 1.0f / sqrtf(z2); break;
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