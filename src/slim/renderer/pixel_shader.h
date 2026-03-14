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
    const RenderState& render_state;
    Color pixel;

    vec3 P, N, L, LP, V, R;
    f32 NdotL, NdotV, roughness;
    BRDFType brdf;

    INLINE_XPU const Color& shade(
        const GroundHit& ground_hit,
        const WallHit& wall_hit,
        const Slice<TileEdge>& edges,
        const Slice<Circle>& columns,
        const vec2& position,
        u16 y,
        i32 mid_point) {
        pixel = Magenta;

        if (!wall_hit.isValid()) return pixel;

        vec3 Ro;
        f32 u, v;
        u8 mip_level, texture_id, edge_is;

        const vec2 ray_hit_position = position + wall_hit.ray_direction * ground_hit.z;
        if (y < wall_hit.top ||
            y > wall_hit.bot) {
            const bool is_ceiling = y < mid_point;
            const vec2 start = 1.0f;
            const vec2 end = {
                (f32)(settings.tile_map_width - 1),
                (f32)(settings.tile_map_height - 1)
            };
            if (!inRange(start, ray_hit_position, end)) return pixel;

            mip_level = ground_hit.mip;
            v = ray_hit_position.y - (f32)(i32)ray_hit_position.y;
            u = ray_hit_position.x - (f32)(i32)ray_hit_position.x;
            edge_is = is_ceiling ? ABOVE : BELOW;
            texture_id = is_ceiling ? settings.ceiling_texture_id : settings.floor_texture_id;
            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;
            P.x = ray_hit_position.x;
            P.z = ray_hit_position.y;
            P.y = is_ceiling ? 1.0f : -1.0f;
            // u *= 0.5f;
            // v *= 0.5f;
        } else {
            mip_level = wall_hit.mip;
            v = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
            u = wall_hit.u;
            edge_is = wall_hit.edge_is;
            texture_id = wall_hit.texture_id;
            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;
            P.x = wall_hit.hit_position.x;
            P.z = wall_hit.hit_position.y;
            P.y = (1.0f - v) * 2.0f - 1.0f;
            v *= 2.0f;
            v -= (f32)(i32)v;
        }

        if (render_state.render_mode == RenderMode_Beauty ||
            render_state.render_mode == RenderMode_Color)
            pixel = settings.textures[texture_id].mips[mip_level].sampleColor(u, v);
        else if (render_state.render_mode == RenderMode_Light)
            pixel = White;

        roughness = 1.0f;
        if (render_state.flags & USE_ROUGHNESS_MAP &&
            (render_state.render_mode == RenderMode_Roughness ||
             render_state.render_mode == RenderMode_Beauty ||
             render_state.render_mode == RenderMode_Light)) {
            roughness = settings.textures[texture_id + 1].mips[mip_level].sampleColor(u, v).r;
        }
        N = {0.0f, 0.0f, 1.0f};
        const bool normalNeeded = render_state.render_mode == RenderMode_Beauty ||
                                  render_state.render_mode == RenderMode_Normal ||
                                  render_state.render_mode == RenderMode_Light;
        if (normalNeeded && (render_state.flags & USE_NORMAL_MAP))
            N = vec3{settings.textures[texture_id + 2].mips[mip_level].sampleColor(u, v)}.scaleAdd(2.0f, -1.0f).normalized();
        f32 bump = 0.0f;
        f32 portal_distance_fraction = 0.0f;
        const Portal* portal = nullptr;
        if (!(edge_is == ABOVE || edge_is == BELOW)) {
            vec3 PortalToP;
            if (render_state.portal_from.edge_id != INVALID_EDGE_ID &&
                render_state.portal_from.edge_is == edge_is) {
                portal = &render_state.portal_from;
                PortalToP = vec3{P.x, P.y * 0.5f, P.z} - portal->position;
                portal_distance_fraction = PortalToP.squaredLength();
                if (portal_distance_fraction > (portal->radius * portal->radius))
                    portal = nullptr;
            }

            if (!portal &&
                render_state.portal_to.edge_id != INVALID_EDGE_ID &&
                render_state.portal_to.edge_is == edge_is) {
                portal = &render_state.portal_to;
                PortalToP = vec3{P.x, P.y * 0.5f, P.z} - portal->position;
                portal_distance_fraction = PortalToP.squaredLength();
                if (portal_distance_fraction > (portal->radius * portal->radius))
                    portal = nullptr;
            }

            if (portal) {
                portal_distance_fraction = sqrt(portal_distance_fraction);
                PortalToP /= portal_distance_fraction;
                portal_distance_fraction = 1.0f - portal_distance_fraction / portal->radius;
                if (normalNeeded && portal_distance_fraction < ((0.4f * (3.0f / 4.0f)))) {
                    if      (edge_is & FACING_DOWN ) PortalToP = {  PortalToP.x,  PortalToP.y, 1.0f - sqrt(PortalToP.x*PortalToP.x + PortalToP.y*PortalToP.y)};
                    else if (edge_is & FACING_UP   ) PortalToP = {  -PortalToP.x, PortalToP.y, sqrt(PortalToP.x*PortalToP.x + PortalToP.y*PortalToP.y) - 1.0f};
                    else if (edge_is & FACING_LEFT ) PortalToP = {-PortalToP.z,   PortalToP.y, 1.0f - sqrt(PortalToP.z*PortalToP.z + PortalToP.y*PortalToP.y)};
                    else if (edge_is & FACING_RIGHT) PortalToP = { PortalToP.z,   PortalToP.y, sqrt(PortalToP.z*PortalToP.z + PortalToP.y*PortalToP.y) - 1.0f};

                    bump = portal_distance_fraction / (0.4f * (3.0f / 4.0f));
                    if (bump < 0.5f) {
                        if (bump < 0.25f)
                            bump = smoothStep(0.0f, 1.0f, bump / 0.25f);
                        else
                            bump = 1.0f - smoothStep(0.0f, 1.0f, (bump - 0.25f) / 0.25f);

                        N += PortalToP * bump;
                    } else {
                        if (bump < 0.75f)
                            bump = smoothStep(0.0f, 1.0f, (bump - 0.5f) / 0.25f);
                        else
                            bump = 1.0f - smoothStep(0.0f, 1.0f, (bump - 0.75f) / 0.25f);

                        N -= PortalToP * bump;
                    }
                    N = N.normalized();
                }
            }
        }

        if (normalNeeded) {
            if      (edge_is & FACING_DOWN ) N = {   N.x,   N.y,    N.z};
            else if (edge_is & FACING_UP   ) N = {  -N.x,   N.y,   -N.z};
            else if (edge_is & FACING_LEFT ) N = {-N.z,   N.y,  N.x};
            else if (edge_is & FACING_RIGHT) N = { N.z,   N.y, -N.x};
            else if (edge_is & ABOVE)        N = {   N.x,-N.z, -N.y};
            else if (edge_is & BELOW)        N = {   N.x, N.z, -N.y};
            else if (edge_is == 0 && wall_hit.column_id != INVALID_COLUMN_ID) {
                mat3 m{vec3{wall_hit.hit_normal.y, 0.0f, -wall_hit.hit_normal.x},
                       vec3{0.0f, 1.0f, 0.0f},
                        vec3{wall_hit.hit_normal.x, 0.0f, wall_hit.hit_normal.y}};
                N = m * N;
            }
        }

        float AO = 1.0f;
        if (render_state.flags & USE_AO_MAP &&
            (render_state.render_mode == RenderMode_AO ||
             render_state.render_mode == RenderMode_Beauty ||
             render_state.render_mode == RenderMode_Light)) {
            AO = settings.textures[texture_id + 3].mips[mip_level].sampleColor(u, v).r;
            AO *= AO;
            AO *= AO;
        }

        switch (render_state.render_mode) {
            case RenderMode_Color: break;
            case RenderMode_AO: pixel = AO; break;
            case RenderMode_UVs: pixel = Color(u, v, 0); break;
            case RenderMode_Depth: pixel = 1.0f / (Ro - P).length(); break;
            case RenderMode_Normal: pixel = N.scaleAdd(0.5, 0.5f).asColor(); break;
            case RenderMode_MipLevel: pixel = Color(settings.mip_level_colors[mip_level]); break;
            case RenderMode_Roughness: pixel = roughness; break;
            case RenderMode_Untextured: pixel = Color(
                edge_is == ABOVE ?
                    settings.untextured_ceiling_color :
                    (edge_is == BELOW ?
                        settings.untextured_floor_color :
                        settings.untextured_wall_color)); break;
            default: {
                Ray ray;
                TileEdge edge;
                Color light = Black;
                Color flare = Black;
                Color portal_glow = Black;
                if (portal) {
                    if (portal_distance_fraction > 0.5f)
                        pixel *= portal->color;
                    else if (portal_distance_fraction > 0.4f)
                        pixel *= Color(White).lerpTo(portal->color, smoothStep(0.0f, 1.0f, (portal_distance_fraction - 0.4f) / 0.1f));
                    else if (portal_distance_fraction > 0.3f)
                        portal_glow = 0.5f * portal->color.lerpTo(Color(Black), smoothStep(0.0f, 1.0f, (portal_distance_fraction - 0.3f) / 0.1f));
                    else if (portal_distance_fraction > 0.2f)
                        portal_glow = 0.5f * Color(Black).lerpTo(portal->color, smoothStep(0.0f, 1.0f, (portal_distance_fraction - 0.2f) / 0.1f));
                }

                brdf = (BRDFType)(render_state.flags & BRDF_MASK);
                V = (Ro - P).normalized();
                if (brdf == BRDF_GGX) NdotV = clampedValue(N.dot(V));
                else if (brdf == BRDF_Phong) R = (-V).reflectedAround(N);

                for (u8 i = 0; i < render_state.light_count; i++) {
                    const PointLight& point_light{render_state.lights[i]};
                    L = point_light.position - P;

                    if (render_state.flags & CAST_SHADOWS) {
                        vec2 L2d{L.x, L.z};
                        const f32 distance_2d_squared = L2d.squaredLength();
                        ray.update(vec2{P.x, P.z}, L2d / sqrtf(distance_2d_squared));
                        f32 closest_hit_distance = 1000000.0f;
                        for (u16 e = 0; e < (u16)edges.size; e++) {
                            edge = edges.data[e];
                            if (edge.isVisible(ray.origin) && ray.intersectsWithEdge(edge)) {
                                ray.hit.distance = (ray.hit.position - ray.origin).squaredLength();
                                if (ray.hit.distance < closest_hit_distance)
                                    closest_hit_distance = ray.hit.distance;
                            }
                        }

                        ray.hit.distance  = closest_hit_distance;

                        for (u8 c = 0; c < (u8)columns.size; c++)
                            ray.intersectsWithCircle(columns[c]);

                        if (ray.hit.distance < distance_2d_squared)
                            continue;
                    }

                    f32 attenuation = 1.0f / L.squaredLength();
                    L *= sqrtf(attenuation);

                    // if (i == 0) attenuation *= attenuation;
                    const f32 Li = point_light.intensity * attenuation;
                    NdotL = clampedValue(N.dot(L));

                    f32 Fs = 0.0f;
                    f32 F = 0.0f;

                    if (brdf == BRDF_GGX) {
                        const vec3 H = (L + V).normalized();
                        const f32 NdotH = clampedValue(N.dot(H));
                        F = schlickFresnel(clampedValue(H.dot(L)), 0.04f);
                        Fs = GGX(roughness, NdotL, NdotV, NdotH);
                    } else if (brdf != BRDF_Lambert) {
                        F = roughness;
                        f32 exponent = 16.0f;
                        f32 specular_factor = 0.0f;
                        if (brdf == BRDF_Phong) {
                            exponent = 4.0f;
                            specular_factor = clampedValue(R.dot(L));
                        } else { // BLINN
                            specular_factor = clampedValue(N.dot((L + V).normalized()));
                        }
                        if (specular_factor > 0.0f)
                            Fs = powf(specular_factor, exponent);
                    }

                    light += point_light.color * Li * NdotL * lerp(Fs, ONE_OVER_PI, F);

                    if (i) {
                        const vec3 RoL = Ro - point_light.position;
                        // if (!(edge_is == ABOVE || edge_is == BELOW)) V.y *= 0.5f;
                        const f32 distance = (V * V.dot(RoL) - RoL).length();
                        if (distance < settings.projectile_radius) {
                            f32 flare_intensity = 1.0f - ((distance) / (settings.projectile_radius));

                            flare += point_light.color * (0.2f*point_light.intensity * powf(flare_intensity, 8));
                        }
                    }
                }

                pixel *= light + AO * 0.01f * render_state.lights[0].color;
                pixel += flare + portal_glow;
            }
        }

        if (render_state.flags & EDITING_WALLS &&
            ((i32)ray_hit_position.x == (i32)render_state.hovered_pos.x &&
             (i32)ray_hit_position.y == (i32)render_state.hovered_pos.y) ||
            render_state.flags & EDITING_COLUMNS &&
            (ray_hit_position - render_state.hovered_pos).squaredLength() < 0.01f) {
            Color selection{-0.02f};
            if (render_state.flags & EDITING_WALLS)
                selection.g = -selection.g;
            else
                selection.r = -selection.r;

            pixel += selection;
            pixel.r = clampedValue(pixel.r, 0.0f, 1.0f);
            pixel.g = clampedValue(pixel.g, 0.0f, 1.0f);
            pixel.b = clampedValue(pixel.b, 0.0f, 1.0f);
        }

        return pixel;
    }
};