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

INLINE_XPU f32 ggxSchlickSmith_G(f32 roughness, f32 NdotL, f32 NdotV) {
    f32 a = roughness * roughness;
    f32 k = a * 0.5f; // Approximation from Karis (UE4)
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
}

struct SphereTracer {
    vec3 RoC;
    f32 proj, c, t_near, t_far, t_max;

    INLINE_XPU bool prepare(const vec3 &center, const vec3 &Ro, const vec3 &Rd) {
        RoC = center - Ro;
        proj = Rd.dot(RoC);
        return proj > 0.0f;
    }

    INLINE_XPU bool hit(f32 radius, f32 one_over_radius, f32 &max_distance) {
        t_max = max_distance * one_over_radius;
        proj *= one_over_radius;
        c = (RoC * one_over_radius).squaredLength() - 1;
        f32 h = proj*proj - c;

        if (h >= 0) {
            h = sqrtf(h);
            t_near = proj - h;
            t_far  = proj + h;
            if (t_far > 0 && t_near < t_max) {
                max_distance = t_max * radius;
                return true;
            }
        }

        return false;
    }

    INLINE_XPU f32 integrateDensity() {
        f32 tn = Max(t_near, 0);
        f32 tf = Min(t_far, t_max);
        f32 tn2 = tn*tn;
        f32 tf2 = tf*tf;
        return (c*tn - proj*tn2 + tn*tn2/3.0f - (
                c*tf - proj*tf2 + tf*tf2/3.0f   ))
                * (3.0f / 4.0f);
    }
};

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


enum struct HitKind : u8 {
    Floor,
    Ceiling,
    FlatWall,
    RoundedCorner,
    Column
};


struct PixelShader {
    const RenderData& render_data;
    const RenderState& render_state;
    const Slice<TileEdge>& edges;
    const Slice<Circle>& columns;
    const Portals& portals;

    PortalRenderData portal_render_data;
    SphereTracer sphere_tracer;
    Ray ray;
    RayHit hit;
    const Portal* portal;
    Color flare_light;
    vec3 P, N, L, V, R, Ro, PortalToP, hitN, hitP;
    vec2 ground_hit_position, L2d;
    f32 NdotL, NdotV, roughness, u, v, hit_distance, parallax_occlusion_depth;
    HitKind hit_kind;
    u8 mip_level, texture_id;
    bool use_secondary_hit;

    INLINE_XPU vec2 parallaxViewDir(const WallHit& wall_hit, const vec3& view_dir) {
        vec2 flat_wall;
        switch (wall_hit.facing) {
            case Facing::Down : flat_wall = {   view_dir.x, -view_dir.y}; break;
            case Facing::Up   : flat_wall = {  -view_dir.x, -view_dir.y}; break;
            case Facing::Left : flat_wall = { view_dir.z, -view_dir.y}; break;
            case Facing::Right: flat_wall = {-view_dir.z, -view_dir.y}; break;
        }

        vec2 column{
            wall_hit.hit_normal.perp().dot({view_dir.x, view_dir.z}),
            -view_dir.y
        };

        switch (hit_kind) {
            case HitKind::Column: return column;
            case HitKind::FlatWall: return flat_wall;
            case HitKind::RoundedCorner: return flat_wall.lerpTo(column, wall_hit.corner_fraction);
            default: return { view_dir.x, view_dir.z };
        }
    }

    INLINE_XPU f32 parallaxOcclusionMap(const WallHit& wall_hit, bool check_silhouette = true) {
        const TextureMip& depth_texture{render_data.textures[texture_id + 4].mips[mip_level]};
        f32 curr_depth_map_value = depth_texture.sampleColor(u, v).r;
        if (curr_depth_map_value == 0.0f)
            return 0.0f;

        f32 silhouette_factor = fminf(fabsf(V.x), fabsf(V.z));
        f32 circle_uv_scale = 1.0f;
        f32 corner_fraction = 1.0f;
        vec2 circle_center{};
        switch (hit_kind) {
            case HitKind::Column: {
                circle_center = columns[wall_hit.column_id].position;
                const f32 radius = columns[wall_hit.column_id].radius;
                circle_uv_scale = pi * 2.0f / (3.0f * lerp((f32)((i32)radius + 1), (f32)((i32)radius + 2), fractionOf(radius)));
                silhouette_factor *= lerp(0.75f, 6.0f, radius);
                break;
            }
            case HitKind::RoundedCorner:
                circle_center = wall_hit.hit_center;
                corner_fraction = wall_hit.corner_fraction;
                silhouette_factor *= corner_fraction * render_state.rounded_corners_scale;
                circle_uv_scale = 0.25f * pi * render_state.rounded_corners_scale;
                break;
        }

        // calculate the size of each layer
        const f32 layer_height = 1.0f / lerp(
            (f32)render_state.parallax_occlusion_max_steps,
            (f32)render_state.parallax_occlusion_min_steps,
            NdotV * (1.0f - silhouette_factor));
        const f32 scale = layer_height * render_state.parallax_occlusion_scale;
        vec2 d_uv = parallaxViewDir(wall_hit, V) * scale * lerp(1.0f, circle_uv_scale, corner_fraction);

        vec2 P2d{P.x, P.z};
        const vec2 V2d{V.x, V.z};
        const vec2 step_V{V2d * -scale};

        // depth of current layer
        f32 curr_layer_depth = 0.0;

        vec2 uv{u, v};
        vec2 prev_uv;
        f32 prev_depth_map_value = curr_depth_map_value;
        const bool round_corners = hit_kind == HitKind::Column ||
                                   hit_kind == HitKind::RoundedCorner;

        f32 z = NdotV * scale;

        while (curr_layer_depth < curr_depth_map_value)
        {
            prev_uv = uv;
            prev_depth_map_value = curr_depth_map_value;

            // shift texture coordinates along direction
            uv -= d_uv;
            if (round_corners) {
                P2d += step_V;
                vec2 N2d{(P2d - circle_center).normalized()};

                if (check_silhouette) {
                    const f32 N2DdotV = N2d.dot(V2d);
                    z += N2DdotV * scale * silhouette_factor * (N2DdotV < 0.0f ? (f32)render_state.parallax_occlusion_max_steps : 1.0f);//0.075f);
                    if (z < 0.0f)
                        return -1.0f;
                }
                if (hit_kind == HitKind::RoundedCorner && wall_hit.rounding == Rounding::Convex)
                    N2d = -N2d;
                d_uv.x = lerp(d_uv.x, N2d.perp().dot(V2d) * scale * circle_uv_scale, corner_fraction);
            }

            // get depthmap value at current texture coordinates
            curr_depth_map_value = depth_texture.sampleColor(uv.x, uv.y).r;

            // get depth of next layer
            curr_layer_depth += layer_height;
        }
        // get depth after and before collision for linear interpolation
        f32 after_depth  = curr_depth_map_value - curr_layer_depth;
        f32 before_depth = prev_depth_map_value - curr_layer_depth + layer_height;

        // interpolation of texture coordinates
        f32 weight = after_depth / (after_depth - before_depth);

        const vec2 new_uv = lerp(uv, prev_uv, weight);
        u = new_uv.x;
        v = new_uv.y;

        return lerp(curr_depth_map_value, prev_depth_map_value, weight);
    }

    INLINE_XPU bool init_base(
        const GroundHit& ground_hit,
        const WallHit& wall_hit,
        const vec2& position,
        u16 y,
        i32 mid_point) {
        ground_hit_position = position + wall_hit.ray_direction * ground_hit.z;
        hitN = 0.0f;
        hit_kind = HitKind::FlatWall;
        if (y < wall_hit.top ||
            y > wall_hit.bot) {
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

            if (y < mid_point) {
                hit_kind = HitKind::Ceiling;
                texture_id = render_data.ceiling_texture_id;
                hitP.y = 1.0f;
                hitN.y = -1.0f;
            } else {
                hit_kind = HitKind::Floor;
                texture_id = render_data.floor_texture_id;
                hitP.y = -1.0f;
                hitN.y = 1.0f;
            }
            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;
            hitP.x = ground_hit_position.x;
            hitP.z = ground_hit_position.y;
        } else {
            mip_level = wall_hit.mip;

            v = wall_hit.v + wall_hit.texel_step * (f32)(y - wall_hit.top);
            u = wall_hit.u;

            // facing = wall_hit.facing;
            texture_id = wall_hit.texture_id;

            Ro.x = position.x;
            Ro.z = position.y;
            Ro.y = 0.0f;

            hitP.x = wall_hit.hit_position.x;
            hitP.z = wall_hit.hit_position.y;
            hitP.y = (1.0f - v) * 2.0f - 1.0f;

            hitN.x = wall_hit.hit_normal.x;
            hitN.z = wall_hit.hit_normal.y;

            v *= 2.0f;
            v -= (f32)(i32)v;

            if (wall_hit.column_id != INVALID_COLUMN_ID)
                hit_kind = HitKind::Column;
            else if (wall_hit.rounding != Rounding::None)
                hit_kind = HitKind::RoundedCorner;
        }

        V = Ro - hitP;
        hit_distance = V.length();
        V /= hit_distance;
        NdotV = hitN.dot(V);
        P = hitP;

        return true;
    }

    INLINE_XPU bool init(
        const GroundHit* ground_hits,
        const WallHitGroup* wall_hit_groups,
        const vec2& origin,
        u16 x,
        u16 y,
        i32 mid_point,
        i32 portal_index) {
        const WallHitGroup* wall_hit_group = &wall_hit_groups[x];
        const WallHit& wall_hit = portal_index == -1 ?
            wall_hit_group->main_hit :
            wall_hit_group->portal_hits[portal_index];

        if (!wall_hit.isValid())
            return false;

        const vec2& position = portal_index == -1 ? origin : wall_hit_group->portal_origins[portal_index];
        const GroundHit& ground_hit{ground_hits[y]};

        if (!init_base(ground_hit, wall_hit, position, y, mid_point))
            return false;

        use_secondary_hit = false;
        parallax_occlusion_depth = 0.0f;
        if (render_state.parallax_occlusion_scale > 0.0f && NdotV > 0.0f) {
            const bool check_silhouette = (hit_kind == HitKind::Column ||
                                          (hit_kind == HitKind::RoundedCorner &&
                                          wall_hit.rounding != Rounding::Concave)) &&
                                          wall_hit_group->secondary_hit.isValid() &&
                                          render_state.rounded_corners_radius > render_state.parallax_occlusion_scale;
            parallax_occlusion_depth = parallaxOcclusionMap(wall_hit, check_silhouette);
            if (parallax_occlusion_depth == -1.0f && portal_index == -1) {
                if (!init_base(ground_hit, wall_hit_group->secondary_hit, position, y, mid_point))
                    return false;
                parallax_occlusion_depth = parallaxOcclusionMap(wall_hit_group->secondary_hit, false);
                if (parallax_occlusion_depth == -1.0f)
                    return false;

                use_secondary_hit = true;
            }
            P = hitP - V * parallax_occlusion_depth * render_state.parallax_occlusion_scale * NdotV;
        }

        return true;
    }

    INLINE_XPU void prepareRayCast() {
        L2d = vec2{L.x, L.z};
        hit.init();
        ray.update(vec2{hitP.x, hitP.z}, L2d, ray.forward);
    }

    INLINE_XPU bool rayHit(
        const Slice<TileEdge>& edges,
        const Slice<Circle>& columns,
        const u16 skip_edge_id = INVALID_EDGE_ID,
        const f32 max_hit_distance = 1000000.0f
        ) {
        f32 closest_hit_distance = max_hit_distance;
        const f32 distance_2d_squared = L2d.squaredLength();
        hit.init();
        for (u16 edge_id = 0; edge_id < (u16)edges.size; edge_id++) {
            if (edge_id == skip_edge_id)
                continue;

            if (ray.intersectsWithEdge(edges.data[edge_id], hit, render_state.rounded_corners_radius)) {
                hit.distance = (hit.position - ray.origin).squaredLength();
                if (hit.distance < closest_hit_distance)
                    closest_hit_distance = hit.distance;
            }
        }

        hit.distance  = closest_hit_distance;

        f32 t;
        for (u8 c = 0; c < (u8)columns.size; c++)
            if (ray.intersectsWithCircle(columns[c].position, columns[c].radius, t) && t > 0.0f && t*t < hit.distance)
                hit.distance = t*t;

        return hit.distance < distance_2d_squared;
    }

    INLINE_XPU bool prepare(
        const GroundHit* ground_hits,
        const WallHitGroup* wall_hit_groups,
        const vec2& origin,
        u16& x,
        u16& y,
        i32 mid_point,
        i32 portal_index = -1) {
        if (!init(ground_hits, wall_hit_groups, origin, x, y, mid_point, portal_index))
            return false;

        flare_light = Black;
        if (render_state.render_mode == RenderMode_Beauty) {
            V = -V;
            const f32 up_aim = (f32)render_state.screen_height * 0.5f - mid_point;
            const vec2 screen_ratio{(f32)x / (f32)render_state.screen_width, ((f32)y + up_aim) / (f32)render_state.screen_height};
            f32 fraction = 1.0f / render_state.step_count;
            Color enemy_light{Black};
            u8 enemy_count =0;
            for (u8 i = 1; i < (render_state.light_count + render_state.enemy_count); i++) {
                const PointLight& point_light{
                    i < render_state.light_count ?
                    render_state.lights[i] :
                    render_state.enemies[i - render_state.light_count]
                };
                bool in_enemy = false;
                if (sphere_tracer.prepare(point_light.position, Ro, V)) {
                    vec3 LP = V * sphere_tracer.proj - sphere_tracer.RoC;
                    f32 distance = LP.squaredLength();
                    if (i < render_state.light_count) {
                        if (sphere_tracer.hit(PROJECTILE_RADIUS, 1.0f / PROJECTILE_RADIUS, hit_distance)) {
                        // if (distance < (PROJECTILE_RADIUS * PROJECTILE_RADIUS)) {
                            f32 flare_intensity = (PROJECTILE_RADIUS - sqrtf(distance)) / PROJECTILE_RADIUS;
                            flare_intensity *= flare_intensity;
                            flare_intensity *= flare_intensity;
                            flare_intensity *= flare_intensity;
                            flare_intensity *= flare_intensity;

                            flare_light += render_state.lights[i].color * render_state.lights[i].intensity * flare_intensity;
                        }
                    } else {
                        vec2 radii = screen_ratio * sphere_tracer.proj;
                        radii.y += point_light.position.y;

                        radii += render_state.time;
                        radii.x *= 9.0f;
                        radii.y *= 5.0f;
                        radii = radii.sin();

                        f32 radius = 0.5f - (radii.x * radii.y) * 0.02f - 0.02f;
                        f32 one_over_radius = 1.0f / radius;
                        if (sphere_tracer.hit(radius, one_over_radius, hit_distance)) {
                            in_enemy = true;

                            LP = Ro - point_light.position;
                            vec3 step = V * radius;
                            vec3 near = LP + step * sphere_tracer.t_near;
                            f32 accum = 0.0f;
                            if (render_state.step_count) {
                                vec3 far = LP + step * sphere_tracer.t_far;
                                step = (far - near) * fraction;
                                LP = near + step * 0.5f;
                                for (u8 k = 0; k < render_state.step_count; k++, LP += step) {
                                    f32 dist = LP.squaredLength();
                                    f32 noise = render_state.noise(LP * (7.0f / sqrtf(dist)));
                                    noise *= noise;
                                    noise *= noise;
                                    noise *= noise;

                                    accum += noise * fraction / (dist * dist);
                                }
                            } else {
                                accum = sphere_tracer.integrateDensity();
                                accum *= accum;
                                accum *= accum;
                                accum *= 16.0f;
                            }

                            distance = sqrtf(distance) * one_over_radius;
                            distance *= distance;
                            distance *= distance;
                            distance *= distance;
                            distance *= distance;
                            distance = 1.0f - distance;

                            enemy_light += 0.35f * Color(-0.25f).lerpTo(point_light.color * (point_light.intensity * accum), distance);
                            enemy_count++;
                            // near = near.normalized();
                            // far = far.normalized();
                            radii += (1.0f + V.dot(near))*3.0f;
                            // radii.x *= render_state.noise(near * 5.0f);
                            // radii.y *= render_state.noise(far * 5.0f);
                            radii *= 9.0f;

                            i32 X = (i32)radii.x + (i32)x;
                            X = X < 0 ? 0 : (X >= render_state.screen_width ? (i32)(render_state.screen_width - 1) : X );
                            x = (u16)X;

                            i32 Y = (i32)radii.y + (i32)y;
                            Y = Y < 0 ? 0 : (Y >= render_state.screen_height ? (i32)(render_state.screen_height - 1) : Y );
                            y = (u16)Y;
                            if (!init(ground_hits, wall_hit_groups, origin, x, y, mid_point, portal_index))
                                return false;
                        }
                    }
                }

                if (!in_enemy && (render_state.flags & VOLUMETRIC) && render_state.step_count && i >= render_state.light_count) {
                    vec3 step = (P - Ro) * fraction;
                    vec3 RP = Ro + step * 0.5f;
                    f32 accum = 0.0f;
                    for (u8 k = 0; k < render_state.step_count; k++, RP += step) {
                        if ((render_state.flags & VOLUMETRIC_SHADOWS)) {
                            L2d = {point_light.position.x - RP.x, point_light.position.z - RP.z};
                            ray.update({RP.x, RP.z}, L2d, ray.forward);
                            if (rayHit(edges, columns))
                                continue;
                        }

                        vec3 RP_local = RP - point_light.position;
                        f32 dist = RP_local.squaredLength();
                        f32 noise = render_state.noise(RP_local * (17.0f / sqrtf(dist)));
                        noise *= noise;
                        noise *= noise;
                        noise *= noise;
                        noise *= noise;

                        accum += noise * 27.0f * fraction / dist;
                    }
                    flare_light += point_light.color * (point_light.intensity * accum);
                }
            }
            if (enemy_count) flare_light +=  enemy_light;
            V = -V;
        }

        portal_render_data.init(portals.areBothActive());
        portal = nullptr;


        const u16 wall_hit_edge_id = portal_index == -1 ?
            wall_hit_groups[x].main_hit.edge_id :
            wall_hit_groups[x].portal_hits[portal_index].edge_id;
        if (hit_kind == HitKind::FlatWall) {
            const Portal* other_portal;
            portal = portals.getPortalsFromWallPosition3D(P, wall_hit_edge_id, &other_portal);
            if (portal) {
                PortalToP = vec3{P.x, P.y * 0.5f, P.z} - vec3{portal->position.x, portal->position.y * 0.5f, portal->position.z};
                const f32 portal_distance = PortalToP.length();
                PortalToP /= portal_distance;

                portal_render_data.update(1.0f - portal_distance / portal->radius);
            }
        }

        return true;
    }

    INLINE_XPU Color render(const WallHit& wall_hit) {
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
            switch (wall_hit.facing) {
                case Facing::Down : PortalToP = {   PortalToP.x, PortalToP.y, 1.0f - sqrt(PortalToP.x*PortalToP.x + PortalToP.y*PortalToP.y)}; break;
                case Facing::Up   : PortalToP = {  -PortalToP.x, PortalToP.y, sqrt(PortalToP.x*PortalToP.x + PortalToP.y*PortalToP.y) - 1.0f}; break;
                case Facing::Left : PortalToP = {-PortalToP.z, PortalToP.y, 1.0f - sqrt(PortalToP.z*PortalToP.z + PortalToP.y*PortalToP.y)}; break;
                case Facing::Right: PortalToP = { PortalToP.z, PortalToP.y, sqrt(PortalToP.z*PortalToP.z + PortalToP.y*PortalToP.y) - 1.0f}; break;
            }

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

        if (normalNeeded)
            switch (hit_kind) {
                case HitKind::Column:
                case HitKind::RoundedCorner: N = mat3{
                    vec3{wall_hit.hit_normal.y, 0.0f, -wall_hit.hit_normal.x},
                           vec3{0.0f, 1.0f, 0.0f},
                            vec3{wall_hit.hit_normal.x, 0.0f, wall_hit.hit_normal.y}
                    } * N; break;
                case HitKind::Ceiling: N = {   N.x,-N.z, -N.y}; break;
                case HitKind::Floor  : N = {   N.x, N.z, -N.y}; break;
                case HitKind::FlatWall:
                    switch (wall_hit.facing) {
                        case Facing::Down : N = {   N.x,   N.y,    N.z}; break;
                        case Facing::Up   : N = {  -N.x,   N.y,   -N.z}; break;
                        case Facing::Left : N = {-N.z,   N.y,  N.x}; break;
                        case Facing::Right: N = { N.z,   N.y, -N.x}; break;
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
                hit_kind == HitKind::Ceiling ?
                    UNTEXTURED_CEILING_COLOR :
                    (hit_kind == HitKind::Floor ?
                        UNTEXTURED_FLOOR_COLOR :
                        UNTEXTURED_WALL_COLOR)); break;
            default: {
                Color light = Black;
                if (portal)
                    portal_render_data.updatePixel(pixel, portal->color);

                // brdf = (BRDFType)(render_state.flags & BRDF_MASK);
                // if (brdf == BRDF_GGX)
                NdotV = clampedValue(N.dot(V));
                // else if (brdf == BRDF_Phong)
                    // R = (-V).reflectedAround(N);

                u8 iterations = portal_render_data.has_both ? 3 : 1;
                for (u8 j = 0; j < iterations; j++) {
                    for (u8 i = 0; i < (render_state.light_count + render_state.enemy_count); i++) {
                        const vec3& point_light_position{
                            i < render_state.light_count ?
                            render_state.lights[i].position :
                            render_state.enemies[i - render_state.light_count].position};
                        u16 light_portal_edge_id = INVALID_EDGE_ID;
                        if (i < render_state.light_count && j > 0) {
                            L = (j == 2 ? render_state.lights_through_portal_to[i] : render_state.lights_through_portal_from[i]) - P;
                            const Portal& light_portal{j == 1 ? portals.to : portals.from};
                            light_portal_edge_id = light_portal.edge_id;

                            prepareRayCast();
                            bool light_ray_goes_through_light_portal = false;
                            if (ray.intersectsWithEdge(edges.data[light_portal.edge_id], hit, render_state.rounded_corners_radius)) {
                                vec3 P2 = P + L * ((hit.position - ray.origin).length() / L2d.length());
                                P2.y *= 0.5f;
                                vec3 PortalToP2 = P2 - vec3{light_portal.position.x, light_portal.position.y*0.5f, light_portal.position.z};
                                if (PortalToP2.squaredLength() < (light_portal.radius * light_portal.radius))
                                    light_ray_goes_through_light_portal = true;
                            }
                            if (!light_ray_goes_through_light_portal)
                                continue;
                        } else L = point_light_position - P;

                        if (render_state.flags & CAST_SHADOWS) {
                            if (j == 0)
                                prepareRayCast();
                            else
                                hit.init();

                            if (rayHit(edges, columns, j > 0 ? light_portal_edge_id : INVALID_EDGE_ID))
                                continue;
                        }

                        f32 attenuation = 1.0f / L.squaredLength();
                        L *= sqrtf(attenuation);

                        f32 shadow_opacity = 1.0f;
                        if (render_state.flags & CAST_SHADOWS) {
                            // number of depth layers
                            if (hitN.dot(L) < 0.0f)
                                continue;

                            if (parallax_occlusion_depth > 0.0f) {
                                const TextureMip& depth_texture{render_data.textures[texture_id + 4].mips[0]};

                                // calculate the size of each layer
                                f32 layer_height = 1.0f / lerp(32.0f, 8.0f, clampedValue(hitN.dot(L)));
                                const vec2 d_uv = parallaxViewDir(wall_hit, L) * layer_height * render_state.parallax_occlusion_scale;
                                vec2 uv = {u, v};

                                // f32 sh0 = parallax_occlusion_depth;
                                // uv = vec2{u, v} + d_uv * 0.88f;
                                // f32 shA = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.88f) * 1.0f;
                                // uv = vec2{u, v} + d_uv * 0.77f;
                                // f32 sh9 = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.77f) * 2.0f;
                                // uv = vec2{u, v} + d_uv * 0.66f;
                                // f32 sh8 = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.66f) * 4.0f;
                                // uv = vec2{u, v} + d_uv * 0.55f;
                                // f32 sh7 = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.55f) * 6.0f;
                                // uv = vec2{u, v} + d_uv * 0.44f;
                                // f32 sh6 = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.44f) * 8.0f;
                                // uv = vec2{u, v} + d_uv * 0.33f;
                                // f32 sh5 = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.33f) * 10.0f;
                                // uv = vec2{u, v} + d_uv * 0.22f;
                                // f32 sh4 = -(depth_texture.sampleColor(uv.x, uv.y).r - sh0 - 0.22f) * 12.0f;
                                // shadow_opacity = fmaxf( fmaxf( fmaxf( fmaxf( fmaxf( fmaxf( shA, sh9 ), sh8 ), sh7 ), sh6 ), sh5 ), sh4 );
                                // shadow_opacity = shadow_opacity * 0.25f;
                                // shadow_opacity = 1 - shadow_opacity;

                                f32 curr_depth_map_value = parallax_occlusion_depth;

                                f32 curr_layer_depth = curr_depth_map_value;
                                f32 prev_layer_depth = curr_depth_map_value;
                                f32 prev_depth_map_value = curr_depth_map_value;

                                while (prev_layer_depth <= prev_depth_map_value && prev_layer_depth > 0.0)
                                {
                                    uv += d_uv;
                                    prev_layer_depth = curr_layer_depth;
                                    prev_depth_map_value = curr_depth_map_value;
                                    curr_depth_map_value = depth_texture.sampleColor(uv.x, uv.y).r;
                                    curr_layer_depth -= layer_height;
                                }

                                if (prev_layer_depth > prev_depth_map_value) {
                                    if (curr_layer_depth > curr_depth_map_value)
                                        shadow_opacity = 0.25f;
                                    else {
                                        f32 after_depth  = curr_depth_map_value - curr_layer_depth;
                                        // if (after_depth > 0.0f) {
                                            f32 before_depth = prev_depth_map_value - curr_layer_depth - layer_height;

                                            // interpolation of texture coordinates
                                            shadow_opacity = 0.25f + 0.75f * after_depth / (after_depth - before_depth);
                                        // }
                                    }
                                }
                            }
                        }

                        const f32 Li = shadow_opacity * attenuation * (
                            i < render_state.light_count ?
                            render_state.lights[i].intensity :
                            render_state.enemies[i - render_state.light_count].intensity);
                        NdotL = clampedValue(N.dot(L));

                        f32 Fs = 0.0f;
                        f32 F = 0.0f;

                        if (render_state.brdf == BRDF_GGX) {
                            const vec3 H = (L + V).normalized();
                            const f32 NdotH = clampedValue(N.dot(H));
                            F = schlickFresnel(clampedValue(H.dot(L)), 0.04f);
                            Fs = GGX(roughness, NdotL, NdotV, NdotH);
                        } else if (render_state.brdf != BRDF_Lambert) {
                            F = roughness;
                            f32 exponent = 16.0f;
                            f32 specular_factor = 0.0f;
                            if (render_state.brdf == BRDF_Phong) {
                                exponent = 4.0f;
                                specular_factor = clampedValue(R.dot(L));
                            } else { // BLINN
                                specular_factor = clampedValue(N.dot((L + V).normalized()));
                            }
                            if (specular_factor > 0.0f)
                                Fs = powf(specular_factor, exponent);
                        }

                        Color color = render_state.lights[i].color;
                        if (i >= render_state.light_count) {
                            color = render_state.enemies[i - render_state.light_count].color;
                            vec3 LP = P - point_light_position;
                            f32 dist = LP.squaredLength();
                            f32 noise = render_state.noise(LP * (17.0f / sqrtf(dist)));
                            noise *= noise;
                            noise *= noise;
                            noise *= noise;
                            noise = 2.0f * noise - 1.0f;
                            noise = 29.0f + 27.0f * noise;
                            color *= noise;
                        }
                        light += color * Li * NdotL * lerp(Fs, ONE_OVER_PI, F);
                    }
                }

                pixel *= light + AO * 0.01f * render_state.lights[0].color;
            }
        }

        if (render_state.edit == Edit::Walls &&
            ((i32)ground_hit_position.x == (i32)render_state.hovered_pos.x &&
             (i32)ground_hit_position.y == (i32)render_state.hovered_pos.y) ||
            (render_state.edit == Edit::Columns || render_state.edit == Edit::Enemies) &&
            (ground_hit_position - render_state.hovered_pos).squaredLength() < (render_state.edit == Edit::Enemies ? 0.15f : 0.02f)) {
            pixel = pixel.lerpTo(Color(render_state.edit == Edit::Enemies ? Magenta : (render_state.edit == Edit::Walls ? Cyan : Yellow)), 0.02f);
        }

        return pixel;
    }

    INLINE_XPU Color shade(
        const GroundHit* ground_hits,
        const WallHitGroup* wall_hit_groups,
        const vec2& position,
        u16 x,
        u16 y,
        i32 mid_point) {
        if (!prepare(ground_hits, wall_hit_groups, position, x, y, mid_point))
            return Magenta;

        const WallHitGroup& wall_hit_group{wall_hit_groups[x]};

        Color pixel = Black;
        if (!(portal_render_data.render_through && !portal_render_data.blend))
            pixel = render(use_secondary_hit ? wall_hit_group.secondary_hit : wall_hit_group.main_hit);

        if (portal_render_data.render_through) {
            f32 portal_blend_factor = portal_render_data.blend_factor;
            f32 pixel_blend_factor = 1.0f - portal_blend_factor;
            pixel *= pixel_blend_factor;

            PixelShader pixel_shader_through_portal{render_data, render_state, edges, columns, portals};
            pixel_shader_through_portal.portal_render_data = portal_render_data;
            for (u8 p = 0; p < MAX_PORTAL_DEPTH && pixel_shader_through_portal.portal_render_data.render_through; p++) {
                const WallHit& portal_wall_hit{wall_hit_group.portal_hits[p]};
                if (!portal_wall_hit.isValid())
                    break;

                if (!pixel_shader_through_portal.prepare(ground_hits, wall_hit_groups, position, x, y, mid_point, p)) {
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

                    pixel += pixel_blend_factor * pixel_shader_through_portal.render(portal_wall_hit);
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