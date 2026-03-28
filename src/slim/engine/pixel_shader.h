#pragma once

#include "./raycast.h"
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


struct PortalRenderData {
    f32 blend_factor;
    f32 distance_fraction;
    bool blend;
    bool has_both;
    bool render_through;

    INLINE_XPU void init(const bool both) {
        render_through = false;
        blend = false;
        blend_factor = 0.0f;
        has_both = both;
        distance_fraction = 0.0f;
    }

    INLINE_XPU void update(const f32 fraction) {
        distance_fraction = fraction;
        render_through = has_both && 0.3f < distance_fraction;
        blend = render_through && distance_fraction < 0.4f;
        blend_factor = render_through ? (blend ? smoothStep(0.0f, 1.0f, (distance_fraction - 0.3f) / 0.1f) : 1.0f) : 0.0f;
    }

    INLINE_XPU void updatePixel(Color& pixel, const Color& portal_color) {
        if (distance_fraction > 0.4f)
            pixel *= portal_color;
        else if (distance_fraction > 0.3f) {
            if (!has_both)
                pixel *= Color(White).lerpTo(portal_color, blend_factor);
            pixel += 0.5f * portal_color.lerpTo(Color(Black), smoothStep(0.0f, 1.0f, (distance_fraction - 0.3f) / 0.1f));
        }
        else if (distance_fraction > 0.2f)
            pixel += 0.5f * Color(Black).lerpTo(portal_color, smoothStep(0.0f, 1.0f, (distance_fraction - 0.2f) / 0.1f));
    }
};


struct PixelShader {
    const RenderData& render_data;
    const RenderState& render_state;
    PortalRenderData portal_render_data;
    const Portal* portal;
    BRDFType brdf;
    Color flare_light;
    vec3 P, N, L, LP, V, R, Ro, PortalToP;
    vec2 ground_hit_position;
    f32 NdotL, NdotV, roughness, u, v;
    u8 mip_level, texture_id, edge_is;

    INLINE_XPU bool prepare(
        const GroundHit& ground_hit,
        const WallHit& wall_hit,
        const Portals& portals,
        const vec2& position,
        u16 y,
        i32 mid_point) {

        ground_hit_position = position + wall_hit.ray_direction * ground_hit.z;
        if (y < wall_hit.top ||
            y > wall_hit.bot) {
            const bool is_ceiling = y < mid_point;
            const vec2 start = 1.0f;
            const vec2 end = {
                (f32)(render_data.map_width - 1),
                (f32)(render_data.map_height - 1)
            };
            if (!inRange(start, ground_hit_position, end))
                return false;

            mip_level = ground_hit.mip;
            v = ground_hit_position.y - (f32)(i32)ground_hit_position.y;
            u = ground_hit_position.x - (f32)(i32)ground_hit_position.x;
            edge_is = is_ceiling ? ABOVE : BELOW;
            texture_id = is_ceiling ? render_data.ceiling_texture_id : render_data.floor_texture_id;
            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;
            P.x = ground_hit_position.x;
            P.z = ground_hit_position.y;
            P.y = is_ceiling ? 1.0f : -1.0f;
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

        flare_light = Black;
        if (render_state.render_mode == RenderMode_Beauty) {
            V = (Ro - P).normalized();
            for (u8 i = 1; i < render_state.light_count; i++) {
                const PointLight& point_light{render_state.lights[i]};
                const vec3 RoL = Ro - point_light.position;
                if (RoL.dot(V) > 0.0f) {
                    f32 distance = (V * V.dot(RoL) - RoL).squaredLength();
                    if (distance < (PROJECTILE_RADIUS * PROJECTILE_RADIUS)) {
                        f32 flare_intensity = (PROJECTILE_RADIUS - sqrtf(distance)) / PROJECTILE_RADIUS;
                        flare_intensity *= flare_intensity;
                        flare_intensity *= flare_intensity;
                        flare_intensity *= flare_intensity;
                        flare_intensity *= flare_intensity;
                        flare_light += point_light.color * point_light.intensity * flare_intensity;
                    }
                }
            }
        }

        portal_render_data.init(portals.areBothActive());
        portal = nullptr;

        if (wall_hit.edge_id != INVALID_EDGE_ID &&
            edge_is != ABOVE &&
            edge_is != BELOW) {

            const Portal* other_portal;
            portal = portals.getPortalsFromWallPosition3D(P, wall_hit.edge_id, &other_portal);
            if (portal) {
                PortalToP = vec3{P.x, P.y * 0.5f, P.z} - vec3{portal->position.x, portal->position.y * 0.5f, portal->position.z};
                const f32 portal_distance = PortalToP.length();
                PortalToP /= portal_distance;

                portal_render_data.update(1.0f - portal_distance / portal->radius);
            }
        }

        return true;
    }

    INLINE_XPU Color render(
        const WallHit& wall_hit,
        const Slice<TileEdge>& edges,
        const Slice<Circle>& columns,
        const Portals& portals) {
        Color pixel = Magenta;
        if (render_state.render_mode == RenderMode_Beauty ||
            render_state.render_mode == RenderMode_Color)
            pixel = render_data.textures[texture_id].mips[mip_level].sampleColor(u, v);
        else if (render_state.render_mode == RenderMode_Light)
            pixel = White;

        roughness = 1.0f;
        if (render_state.flags & USE_ROUGHNESS_MAP &&
            (render_state.render_mode == RenderMode_Roughness ||
             render_state.render_mode == RenderMode_Beauty ||
             render_state.render_mode == RenderMode_Light)) {
            roughness = render_data.textures[texture_id + 1].mips[mip_level].sampleColor(u, v).r;
        }
        N = {0.0f, 0.0f, 1.0f};
        const bool normalNeeded = render_state.render_mode == RenderMode_Beauty ||
                                  render_state.render_mode == RenderMode_Normal ||
                                  render_state.render_mode == RenderMode_Light;
        if (normalNeeded && (render_state.flags & USE_NORMAL_MAP))
            N = vec3{render_data.textures[texture_id + 2].mips[mip_level].sampleColor(u, v)}.scaleAdd(2.0f, -1.0f).normalized();

        if (portal && normalNeeded && portal_render_data.distance_fraction < ((0.4f * (3.0f / 4.0f)))) {
            if      (edge_is & FACING_DOWN ) PortalToP = {  PortalToP.x,  PortalToP.y, 1.0f - sqrt(PortalToP.x*PortalToP.x + PortalToP.y*PortalToP.y)};
            else if (edge_is & FACING_UP   ) PortalToP = {  -PortalToP.x, PortalToP.y, sqrt(PortalToP.x*PortalToP.x + PortalToP.y*PortalToP.y) - 1.0f};
            else if (edge_is & FACING_LEFT ) PortalToP = {-PortalToP.z,   PortalToP.y, 1.0f - sqrt(PortalToP.z*PortalToP.z + PortalToP.y*PortalToP.y)};
            else if (edge_is & FACING_RIGHT) PortalToP = { PortalToP.z,   PortalToP.y, sqrt(PortalToP.z*PortalToP.z + PortalToP.y*PortalToP.y) - 1.0f};

            f32 bump = portal_render_data.distance_fraction / (0.4f * (3.0f / 4.0f));
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
            AO = render_data.textures[texture_id + 3].mips[mip_level].sampleColor(u, v).r;
            AO *= AO;
            AO *= AO;
        }

        switch (render_state.render_mode) {
            case RenderMode_Color: break;
            case RenderMode_AO: pixel = AO; break;
            case RenderMode_UVs: pixel = Color(u, v, 0); break;
            case RenderMode_Depth: pixel = 1.0f / (Ro - P).length(); break;
            case RenderMode_Normal: pixel = N.scaleAdd(0.5, 0.5f).asColor(); break;
            case RenderMode_Roughness: pixel = roughness; break;
            case RenderMode_MipLevel: {
                ColorID colorId;
                switch (mip_level) {
                    case 0: colorId = MIP_LEVEL_0_COLOR; break;
                    case 1: colorId = MIP_LEVEL_1_COLOR; break;
                    case 2: colorId = MIP_LEVEL_2_COLOR; break;
                    case 3: colorId = MIP_LEVEL_3_COLOR; break;
                    case 4: colorId = MIP_LEVEL_4_COLOR; break;
                    case 5: colorId = MIP_LEVEL_5_COLOR; break;
                    case 6: colorId = MIP_LEVEL_6_COLOR; break;
                    case 7: colorId = MIP_LEVEL_7_COLOR; break;
                    case 8: colorId = MIP_LEVEL_8_COLOR; break;
                    default: colorId = Magenta; break;
                }
                pixel = Color(colorId);
                break;
            }
            case RenderMode_Untextured: pixel = Color(
                edge_is == ABOVE ?
                    UNTEXTURED_CEILING_COLOR :
                    (edge_is == BELOW ?
                        UNTEXTURED_FLOOR_COLOR :
                        UNTEXTURED_WALL_COLOR)); break;
            default: {
                Ray ray;
                Color light = Black;
                if (portal)
                    portal_render_data.updatePixel(pixel, portal->color);

                brdf = (BRDFType)(render_state.flags & BRDF_MASK);
                if (brdf == BRDF_GGX) NdotV = clampedValue(N.dot(V));
                else if (brdf == BRDF_Phong) R = (-V).reflectedAround(N);

                u8 iterations = portal_render_data.has_both ? 3 : 1;
                for (u8 j = 0; j < iterations; j++) {
                    for (u8 i = 0; i < render_state.light_count; i++) {
                        const PointLight& point_light{render_state.lights[i]};
                        vec2 L2d;
                        u16 light_portal_edge_id = INVALID_EDGE_ID;
                        if (j > 0) {
                            L = (j == 2 ? render_state.lights_through_portal_to[i] : render_state.lights_through_portal_from[i]) - P;
                            const Portal& light_portal{j == 1 ? portals.to : portals.from};
                            light_portal_edge_id = light_portal.edge_id;
                            L2d = vec2{L.x, L.z};
                            ray.update(vec2{P.x, P.z}, L2d, ray.forward);

                            bool light_ray_goes_through_light_portal = false;
                            if (ray.intersectsWithEdge(edges.data[light_portal.edge_id])) {
                                vec3 P2 = P + L * ((ray.hit.position - ray.origin).length() / L2d.length());
                                P2.y *= 0.5f;
                                vec3 PortalToP2 = P2 - vec3{light_portal.position.x, light_portal.position.y*0.5f, light_portal.position.z};
                                if (PortalToP2.squaredLength() < (light_portal.radius * light_portal.radius))
                                    light_ray_goes_through_light_portal = true;
                            }
                            if (!light_ray_goes_through_light_portal)
                                continue;
                        } else L = point_light.position - P;

                        if (render_state.flags & CAST_SHADOWS) {
                            if (j == 0) {
                                L2d = vec2{L.x, L.z};
                                ray.update(vec2{P.x, P.z}, L2d, ray.forward);
                            } else ray.hit.init();
                            f32 closest_hit_distance = 1000000.0f;

                            const f32 distance_2d_squared = L2d.squaredLength();
                            for (u16 edge_id = 0; edge_id < (u16)edges.size; edge_id++) {
                                if (j > 0 && edge_id == light_portal_edge_id)
                                    continue;

                                if (ray.intersectsWithEdge(edges.data[edge_id])) {
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
                    }
                }

                pixel *= light + AO * 0.01f * render_state.lights[0].color;
            }
        }

        if (render_state.flags & EDITING_WALLS &&
            ((i32)ground_hit_position.x == (i32)render_state.hovered_pos.x &&
             (i32)ground_hit_position.y == (i32)render_state.hovered_pos.y) ||
            render_state.flags & EDITING_COLUMNS &&
            (ground_hit_position - render_state.hovered_pos).squaredLength() < 0.01f) {
            Color selection{-0.02f};
            if (render_state.flags & EDITING_WALLS)
                selection.g = -selection.g;
            else
                selection.r = -selection.r;

            pixel += selection;
        }

        return pixel;
    }

    INLINE_XPU Color shade(
        const GroundHit& ground_hit,
        const WallHitGroup& wall_hit_group,
        const Portals& portals,
        const Slice<TileEdge>& edges,
        const Slice<Circle>& columns,
        const vec2& position,
        u16 y,
        i32 mid_point) {
        if (!wall_hit_group.main_hit.isValid() ||
            !prepare(ground_hit, wall_hit_group.main_hit, portals, position, y, mid_point))
            return Magenta;

        Color pixel = Black;
        if (!(portal_render_data.render_through && !portal_render_data.blend))
            pixel = render(wall_hit_group.main_hit, edges, columns, portals);

        if (portal_render_data.render_through) {
            f32 portal_blend_factor = portal_render_data.blend_factor;
            f32 pixel_blend_factor = 1.0f - portal_blend_factor;
            pixel *= pixel_blend_factor;

            PixelShader pixel_shader_through_portal{render_data, render_state};
            pixel_shader_through_portal.portal_render_data = portal_render_data;
            for (u8 p = 0; p < MAX_PORTAL_DEPTH && pixel_shader_through_portal.portal_render_data.render_through; p++) {
                const WallHit& portal_wall_hit{wall_hit_group.portal_hits[p]};
                if (!portal_wall_hit.isValid())
                    break;

                if (!pixel_shader_through_portal.prepare(ground_hit, portal_wall_hit, portals, wall_hit_group.portal_origins[p], y, mid_point)) {
                    pixel += Color(Magenta) * portal_blend_factor;
                    break;
                }

                flare_light += portal_blend_factor * pixel_shader_through_portal.flare_light;


                if (!(pixel_shader_through_portal.portal_render_data.render_through &&
                    !pixel_shader_through_portal.portal_render_data.blend) ||
                    p == (MAX_PORTAL_DEPTH - 1)) {
                    pixel_blend_factor = portal_blend_factor;
                    if (p < (MAX_PORTAL_DEPTH - 1))
                        pixel_blend_factor *= 1.0f - pixel_shader_through_portal.portal_render_data.blend_factor;

                    pixel += pixel_blend_factor * pixel_shader_through_portal.render(portal_wall_hit, edges, columns, portals);
                }

                portal_blend_factor *= pixel_shader_through_portal.portal_render_data.blend_factor;
            }
        }

        if (render_state.render_mode == RenderMode_Beauty)
            pixel += flare_light;

        pixel.r = clampedValue(pixel.r, 0.0f, 1.0f);
        pixel.g = clampedValue(pixel.g, 0.0f, 1.0f);
        pixel.b = clampedValue(pixel.b, 0.0f, 1.0f);
        return pixel;
    }
};