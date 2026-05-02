#pragma once

#include "renderer.h"

#include "../scene/camera.h"


struct Movable {
    vec3 position, forward;
    f32 spawned_time;

    void init(const vec2 tile_map_position, const vec2 tile_map_forward, const f32 up_aim, const f32 time) {
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


struct Engine {
    TileMap tile_map;
    Camera camera{{0, 0 * DEG_TO_RAD, 0}, {13, 0, 3}};
    Renderer renderer{camera, tile_map.columns, tile_map.edges};

    RenderData& render_data{renderer.render_data};
	RenderState& render_state{renderer.render_state};

    Edit& edit_mode{render_state.edit};

    Movable projectiles[MAX_POINT_LIGHTS];
    Movable enemies[MAX_ENEMIES];
    u8 projectile_count = 0;

    Color torch_light_color{1.0f, 0.6f, 0.35f};
    f32 torch_light_intensity = 4.0f;

    int dragged_enemy_index = -1;
    int dragged_column_index = -1;
    int added_column_index = -1;
    int ray_hit_enemy = -1;

    bool adding = false;
    bool removing = false;
    bool moving = false;

    vec2 edit_offset{0.0f, 0.0f};

    void init() {
        renderer.init();
        uploadEdges(tile_map.edges);
        onMove();
        renderer.onResize();
    }

    void render(u32* window_content) {
        renderer.render(window_content);
    }

    void addLightProjectile(const Color color) {
        Movable& projectile{projectiles[projectile_count++]};
        PointLight& point_light{render_state.lights[render_state.light_count++]};

        projectile.init(renderer.position, renderer.forward, renderer.up_aim, render_state.time);

        point_light.position = projectile.position;
        point_light.color = color;
        point_light.intensity = torch_light_intensity * 0.25f;
    }

    void fireFlare() {
        if (render_state.light_count < (MAX_POINT_LIGHTS - 2))
            addLightProjectile(torch_light_color);
    }

    void launchPortalFrom() {
        renderer.portals.from.projectile_index = projectile_count;
        addLightProjectile(renderer.portals.from.color);
    }

    void launchPortalTo() {
        renderer.portals.to.projectile_index = projectile_count;
        addLightProjectile(renderer.portals.to.color);
    }

    void updateEnemies() {
        static f32 enemy_distances[MAX_ENEMIES];
        for (int i = 0; i < render_state.enemy_count; i++) {
            Movable& enemy{enemies[i]};
            enemy_distances[i] = renderer.forward.dot(vec2{enemy.position.x, enemy.position.z} - renderer.position);
            const f32 elapsed = enemy.spawned_time - render_state.time;

            enemy.position.y = sinf(elapsed) * 0.3f + 0.1f;

            PointLight& enemy_light{render_state.enemies[i]};
            enemy_light.position = enemy.position;
            enemy_light.color = Magenta;
            enemy_light.color *= 0.2f;
            enemy_light.color.r -= sinf(elapsed * 3.0f) * 0.1f + 0.05f;
            enemy_light.color.b -= cosf(elapsed * 2.0f) * 0.05f - 0.1f;
            enemy_light.intensity = 1.0f;
        }
        for (int i = 0; i < render_state.enemy_count; i++) {
            int closest_enemy_index = i;
            for (int j = i + 1; j < render_state.enemy_count; j++) {
                if (enemy_distances[j] < enemy_distances[i])
                    closest_enemy_index = j;
            }
            if (closest_enemy_index != i) {
                Movable closest_enemy{enemies[closest_enemy_index]};
                PointLight closest_enemy_light{render_state.enemies[closest_enemy_index]};
                enemies[closest_enemy_index] = enemies[i];
                enemies[i] = closest_enemy;
                render_state.enemies[closest_enemy_index] = render_state.enemies[i];
                render_state.enemies[i] = closest_enemy_light;
            }
        }
    }

    void updateProjectiles(const f32 delta_time) {
        bool need_generate_wall_hits = false;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(render_data.map_width - 1),
            (f32)(render_data.map_height - 1)
        };
        RayHit hit;
        for (int i = 0; i < projectile_count; i++) {
            Movable& projectile{projectiles[i]};
            const f32 elapsed_time = render_state.time - projectile.spawned_time;
            vec3 projectile_position = projectile.position;
            projectile.updatePosition(delta_time * PROJECTILE_SPEED);

            bool teleported = false;
            bool above_or_below = projectile.position.y >= 1.0f ||
                                  projectile.position.y <= -1.0f;
            bool remove = above_or_below ||
                          !inRange(start, {projectile.position.x, projectile.position.z}, end) ||
                          tile_map.cells[(i32)projectile.position.z][(i32)projectile.position.x].is_full;
            if (remove && !above_or_below) {
                Ray ray;
                vec3 ray_direction_3d = projectile.position - projectile_position;
                vec2 ray_direction_2d = vec2{ray_direction_3d.x, ray_direction_3d.z};
                f32 hit_distance = ray_direction_2d.squaredLength();
                const f32 distance_2d = sqrtf(hit_distance);
                hit.init();
                ray.update(vec2{projectile_position.x, projectile_position.z}, ray_direction_2d / distance_2d, renderer.forward);
                u16 closest_hit_edge_id = INVALID_EDGE_ID;
                Facing closest_hit_edge_facing = Facing::NotApplicable;
                for (u16 edge_id = 0; edge_id < (u16)tile_map.edges.size; edge_id++) {
                    if (ray.intersectsWithEdge(tile_map.edges.data[edge_id], hit, render_state.rounded_corners_radius)) {
                        hit.distance = (hit.position - ray.origin).squaredLength();
                        if (hit.distance < hit_distance) {
                            hit_distance = hit.distance;
                            closest_hit_edge_id = edge_id;
                            closest_hit_edge_facing = hit.facing;
                        }
                    }
                }

                const vec3 projectile_to_edge = ray_direction_3d * (sqrt(hit_distance) / distance_2d);
                const vec3 new_projectile_position = projectile_position + projectile_to_edge;

                const Portal* other_portal = nullptr;
                if (closest_hit_edge_id != INVALID_EDGE_ID) {
                    const Portal* portal = renderer.portals.getPortalsFromWallPosition3D(new_projectile_position, closest_hit_edge_id, &other_portal);

                    if (portal && other_portal->isActive()) {
                        i32 ray_rotation = portal->getRotation(other_portal->facing);

                        vec2 origin{projectile_position.x, projectile_position.z};
                        vec2 origin_to_portal = vec2{portal->position.x, portal->position.z} - origin;
                        vec2 forward{projectile.forward.x, projectile.forward.z};
                        if (ray_rotation == 90) {
                            origin_to_portal = origin_to_portal.ccw90();
                            ray_direction_2d = ray_direction_2d.ccw90();
                            forward = forward.ccw90();
                        } else if (ray_rotation == -90) {
                            origin_to_portal = origin_to_portal.cw90();
                            ray_direction_2d = ray_direction_2d.cw90();
                            forward = forward.cw90();
                        } else if (ray_rotation == 180) {
                            origin_to_portal = -origin_to_portal;
                            ray_direction_2d = -ray_direction_2d;
                            forward = -forward;
                        }
                        const vec2 other_portal_origin = vec2{other_portal->position.x, other_portal->position.z} - origin_to_portal;
                        const vec2 target = other_portal_origin + ray_direction_2d;
                        projectile.position.x = target.x;
                        projectile.position.z = target.y;
                        projectile.forward.x = forward.x;
                        projectile.forward.z = forward.y;
                        remove = false;
                        teleported = true;
                    }
                } else remove = false;
                if (remove && !teleported) {
                    Portal* new_portal = renderer.portals.getPortalsFromProjectileIndex(i, &other_portal);
                    // Check if spawning a portal at the projectile's hit position is allowed:
                    if (new_portal && (
                        !other_portal->isActive() ||
                        (other_portal->position - new_projectile_position).length() > (2 * PORTAL_FINAL_RADIUS))) {
                        new_portal->spawned_time = render_state.time;
                        new_portal->projectile_index = INVALID_PROJECTILE_INDEX;
                        new_portal->position = new_projectile_position;
                        new_portal->edge_id = closest_hit_edge_id;
                        new_portal->facing = closest_hit_edge_facing;
                        new_portal->radius = PORTAL_INITIAL_RADIUS;

                        need_generate_wall_hits = true;
                    }
                }
            }
            if (!remove && !teleported) {
                for (u8 c = 0; c < tile_map.columns.size; c++) {
                    const Circle& column{tile_map.columns.data[c]};
                    const f32 distance_squared = (column.position - vec2{projectile.position.x, projectile.position.z}).squaredLength();
                    if (distance_squared < (column.radius * column.radius)) {
                        remove = true;
                        break;
                    }
                }
            }
            if (remove) {
                projectile_count--;
                render_state.light_count--;
                if (projectile_count == 0) {
                    if (renderer.portals.from.projectile_index == 0)
                        renderer.portals.from.projectile_index = INVALID_EDGE_ID;
                    if (renderer.portals.to.projectile_index == 0)
                        renderer.portals.to.projectile_index = INVALID_EDGE_ID;

                    if (need_generate_wall_hits)
                        renderer.generateWallHits();

                    break;
                }

                if (renderer.portals.from.projectile_index == projectile_count)
                    renderer.portals.from.projectile_index = i;
                if (renderer.portals.to.projectile_index == projectile_count)
                    renderer.portals.to.projectile_index = i;

                projectiles[i] = projectiles[projectile_count];
                render_state.lights[i] = render_state.lights[render_state.light_count];
                i--;
                continue;
            }

            PointLight& point_light{render_state.lights[i + 1]};
            point_light.position = projectile.position;
            if (i == renderer.portals.from.projectile_index)
                point_light.flicker(renderer.portals.from.color, torch_light_intensity * 0.25f, elapsed_time);
            else if (i == renderer.portals.to.projectile_index)
                point_light.flicker(renderer.portals.to.color, torch_light_intensity * 0.25f, elapsed_time);
            else
                point_light.flicker(torch_light_color, torch_light_intensity * 0.25f, elapsed_time);
        }

        if (need_generate_wall_hits)
            renderer.generateWallHits();
    }

    void update(const f32 delta_time) {
        PointLight& torch{render_state.lights[0]};
        torch.position = vec3{renderer.position.x, 0.0f, renderer.position.y};
        torch.position += camera.orientation.X * (sinf(render_state.time*2.7f) * 0.09f + cosf(render_state.time*2.6f) * 0.09f);
        torch.position += camera.orientation.Z * 0.2f;
        torch.position.y += sinf(render_state.time * 2.0f) * 0.3f + 0.1f;

        torch.flicker(torch_light_color, torch_light_intensity, render_state.time);

        renderer.portals.update(render_state.time, tile_map.edges.data);

        if (render_state.enemy_count) updateEnemies();
        if (projectile_count) updateProjectiles(delta_time);
        if (renderer.portals.areBothActive()) renderer.updatePortalLights();
    }

    bool resolveCollisionBetweenCircles(vec2& current, const f32 radius, const Circle& circle) {
        vec2 gap = circle.position - current;
        f32 gap_distance = gap.length();
        if (gap_distance == 0)
            return false;

        f32 min_distance_allowed = radius + circle.radius;
        if (gap_distance < min_distance_allowed) {
            current -= gap * ((min_distance_allowed - gap_distance) / gap_distance);
            return true;
        }
        return false;
        // f32 distance_to_body = gap_distance - radius - circle.radius;
        // if (distance_to_body <= 0.0f)
        //     current += gap * (distance_to_body / gap_distance);
    }
    // bool resolveCollisionBetweenCircles(vec2& current, const f32 radius, const Circle& circle) {
    //     vec2 gap = current - circle.position;
    //     f32 gap_distance = gap.squaredLength();
    //     if (gap_distance == 0)
    //         return false;
    //
    //     gap_distance = sqrtf(gap_distance);
    //     f32 distance_to_body = gap_distance - radius - circle.radius;
    //     if (distance_to_body < 0.0f) {
    //         current += gap * ((0.001f + distance_to_body) / gap_distance);
    //         return true;
    //     }
    //     return false;
    // }

    bool resolveCollisionBetweenCirclesFromInside(vec2& current, const f32 radius, const Circle& circle) {
        vec2 gap = current - circle.position;
        f32 gap_distance = gap.squaredLength();
        if (gap_distance == 0 || gap_distance > (circle.radius * circle.radius))
            return false;

        gap_distance = sqrtf(gap_distance);
        f32 distance_to_body = gap_distance + radius - circle.radius;
        if (distance_to_body > 0.0f) {
            current -= gap * ((0.001f + distance_to_body) / gap_distance);
            return true;
        }
        return false;
    }

    void resolveCollisionsWithRadius(vec2& position, const vec2& movement, const f32 radius, const int skip_column_id = -1, const int skip_enemy_id = -1) {
        const f32 r = render_state.rounded_corners_radius;
        const f32 distance_to_top = position.y - (f32)(i32)(position.y);
        const f32 distance_to_bottom = 1.0f - distance_to_top;
        const f32 distance_to_left = position.x - (f32)(i32)position.x;
        const f32 distance_to_right = 1.0f - distance_to_left;
        bool resolved = false;
        const Tile& tile = tile_map.cells[(i32)(position.y - movement.y)][(i32)(position.x - movement.x)];
        if (!tile.is_full) {
            if (tile.top_left.rounding == Rounding::Concave && distance_to_top < r && distance_to_left < r) {
                Circle concave_corner_circle{{(f32)(i32)position.x + r, (f32)(i32)position.y + r}, r};
                resolved |= resolveCollisionBetweenCirclesFromInside(position, radius, concave_corner_circle);
            }
            if (tile.top_right.rounding == Rounding::Concave && distance_to_top < r && distance_to_right < r) {
                Circle concave_corner_circle{{(f32)((i32)position.x + 1) - r, (f32)(i32)position.y + r}, r};
                resolved |= resolveCollisionBetweenCirclesFromInside(position, radius, concave_corner_circle);
            }
            if (tile.bottom_left.rounding == Rounding::Concave && distance_to_bottom < r && distance_to_left < r) {
                Circle concave_corner_circle{{(f32)(i32)position.x - r, (f32)((i32)position.y + 1) - r}, r};
                resolved |= resolveCollisionBetweenCirclesFromInside(position, radius, concave_corner_circle);
            }
            if (tile.bottom_right.rounding == Rounding::Concave && distance_to_bottom < r && distance_to_right < r) {
                Circle concave_corner_circle{{(f32)((i32)position.x + 1) + r, (f32)((i32)position.y + 1) - r}, r};
                resolved |= resolveCollisionBetweenCirclesFromInside(position, radius, concave_corner_circle);
            }
        }
        if (!resolved) {
            if (movement.x > 0.0f) {
                i32 next_pos = (i32)(position.x + radius);
                const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
                if (next_tile.is_full) {
                    bool resolve_wall_x = true;
                    if (next_tile.top_left.rounding == Rounding::Convex) {
                        if (distance_to_top < r && distance_to_right < radius) {
                            resolve_wall_x = false;
                            Circle convex_corner_circle{{(f32)next_pos + r, (f32)(i32)position.y + r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    } else if (next_tile.bottom_left.rounding == Rounding::Convex) {
                        if (distance_to_bottom < r && distance_to_right < radius) {
                            resolve_wall_x = false;
                            Circle convex_corner_circle{{(f32)next_pos + r, (f32)((i32)position.y + 1) - r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    }
                    if (resolve_wall_x)
                        position.x = (f32)next_pos - radius;
                }
            } else if (movement.x < 0.0f) {
                i32 next_pos = (i32)(position.x - radius);
                const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
                if (next_tile.is_full) {
                    bool resolve_wall_x = true;
                    if (next_tile.top_right.rounding == Rounding::Convex) {
                        if (distance_to_top < r && distance_to_left < radius) {
                            resolve_wall_x = false;
                            Circle convex_corner_circle{{(f32)next_pos + 1.0f - r, (f32)(i32)position.y + r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    } else if (next_tile.bottom_right.rounding == Rounding::Convex) {
                        if (distance_to_bottom < r && distance_to_left < radius) {
                            resolve_wall_x = false;
                            Circle convex_corner_circle{{(f32)next_pos + 1.0f - r, (f32)((i32)position.y + 1) - r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    }
                    if (resolve_wall_x)
                        position.x = (f32)(next_pos + 1) + radius;
                }
            }
        }
        if (!resolved) {
            if (movement.y < 0.0f) {
                i32 next_pos = (i32)(position.y - radius);
                const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
                if (next_tile.is_full) {
                    bool resolve_wall_y = true;
                    if (next_tile.bottom_right.rounding == Rounding::Convex) {
                        if (distance_to_right < r && distance_to_top < radius) {
                            resolve_wall_y = false;
                            Circle convex_corner_circle{{(f32)((i32)position.x + 1) - r, (f32)(next_pos + 1) - r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    } else if (next_tile.bottom_left.rounding == Rounding::Convex) {
                        if (distance_to_left < r && distance_to_top < radius) {
                            resolve_wall_y = false;
                            Circle convex_corner_circle{{(f32)(i32)position.x + r, (f32)(next_pos + 1) - r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    }
                    if (resolve_wall_y)
                        position.y = position.y = (f32)(next_pos + 1) + radius;
                }
            } else if (movement.y > 0.0f) {
                i32 next_pos = (i32)(position.y + radius);
                const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
                if (next_tile.is_full) {
                    bool resolve_wall_y = true;
                    if (next_tile.top_right.rounding == Rounding::Convex) {
                        if (distance_to_right < r && distance_to_bottom < radius) {
                            resolve_wall_y = false;
                            Circle convex_corner_circle{{(f32)((i32)position.x + 1) - r, (f32)next_pos + r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    } else if (next_tile.top_left.rounding == Rounding::Convex) {
                        if (distance_to_left < r && distance_to_bottom < radius) {
                            resolve_wall_y = false;
                            Circle convex_corner_circle{{(f32)(i32)position.x + r, (f32)next_pos + r}, r};
                            resolved |= resolveCollisionBetweenCircles(position, radius, convex_corner_circle);
                        }
                    }
                    if (resolve_wall_y)
                        position.y = (f32)next_pos - radius;
                }
            }
        }
        for (int i = 0; i < (i32)tile_map.columns.size; i++)
            if (i != skip_column_id)
                resolveCollisionBetweenCircles(position, radius, tile_map.columns.data[i]);

        for (int i = 0; i < (i32)render_state.enemy_count; i++)
            if (i != skip_enemy_id)
                resolveCollisionBetweenCircles(position, radius, {{enemies[i].position.x, enemies[i].position.z}, 0.5f});
    }

    void closestHitByRayCast(RayHit& hit, const vec2 origin, const vec2 direction, const bool skip) {
        ray_hit_enemy = -1;
        renderer.castRay(origin, direction, direction, tile_map.edges, tile_map.columns, render_state.rounded_corners_radius, false);
        if (edit_mode == Edit::Walls)
            return;

        f32 t;
        hit.init();
        int skip_column_id = skip ? (edit_mode == Edit::Columns ? (adding ? added_column_index : dragged_column_index) : -1) : -1;
        for (int i = 0; i < (u8)tile_map.columns.size; i++)
            if (i != skip_column_id && renderer.ray.intersectsWithCircle(tile_map.columns[i].position, tile_map.columns[i].radius, t)) {
                if (t < hit.distance) {
                    hit.distance = t;
                    hit.column_id = i;
                }
            }

        int skip_enemy_id = skip ? (edit_mode == Edit::Enemies ? dragged_enemy_index : -1) : -1;
        for (int i = 0; i < render_state.enemy_count; i++)
            if (i != skip_enemy_id &&
                renderer.ray.intersectsWithCircle({enemies[i].position.x, enemies[i].position.z}, 0.5f, t) &&
                t < hit.distance) {
                    hit.distance = t;
                    ray_hit_enemy = i;
                }

        if (ray_hit_enemy != -1) {
            hit.column_id = INVALID_COLUMN_ID;
            hit.edge_id = INVALID_EDGE_ID;
            hit.position = renderer.ray.origin + renderer.ray.direction * hit.distance;
        } else if (hit.column_id != INVALID_COLUMN_ID)
            hit.position = renderer.ray.origin + renderer.ray.direction * hit.distance;
    }

    void displaceByRayCast(vec2& current, const vec2 next, const f32 radius, const bool skip) {
        vec2 movement = next - current;
        const f32 distance = movement.length();
        if (distance == 0.0f)
            return;

        RayHit hit;
        hit.init();
        const vec2 direction = movement / distance;
        closestHitByRayCast(hit, current, direction, skip);

        hit.distance -= 0.001f;
        if (hit.distance < distance)
            movement *= hit.distance / distance;

        current += movement;

        int skip_column_id = skip ? (edit_mode == Edit::Columns ? (adding ? added_column_index : dragged_column_index) : -1) : -1;
        int skip_enemy_id = skip ? (edit_mode == Edit::Enemies ? dragged_enemy_index : -1) : -1;
        resolveCollisionsWithRadius(current, movement, radius, skip_column_id, skip_enemy_id);
    }

    void onMove() {
        vec2 position = vec2(camera.position.x, camera.position.z);
        vec2 movement = position - renderer.position;

        bool teleport = false;
        const Portal* other_portal = nullptr;
        const Portal* portal = nullptr;
        if (renderer.portals.areBothActive() &&
            renderer.castRay(renderer.position, movement, movement, tile_map.edges, tile_map.columns, render_state.rounded_corners_radius, false)) {
            portal = renderer.portals.getPortalsFromWallPosition2D(renderer.closest_hit.position, renderer.closest_hit.edge_id, &other_portal);
            if (portal) {
                if (isHorizontal(renderer.closest_hit.facing))
                    teleport = renderer.position.y > portal->position.z != position.y > portal->position.z;
                else
                    teleport = renderer.position.x > portal->position.x != position.x > portal->position.x;
            }
        }

        if (teleport) {
            i32 rotation = 0;
            position = portal->teleportTo(*other_portal, renderer.position, rotation);
            position += movement.rotatedBy(rotation);

            // Reorient the camera:
            if (     rotation ==  90) camera.orientation.y += DEG_TO_RAD * 90;
            else if (rotation == -90) camera.orientation.y -= DEG_TO_RAD * 90;
            else if (rotation == 180) camera.orientation.y += DEG_TO_RAD * 180;
            const vec2 up = vec2(camera.orientation.Y.x, camera.orientation.Y.z).rotatedBy(rotation);
            const vec2 right = vec2(camera.orientation.X.x, camera.orientation.X.z).rotatedBy(rotation);
            const vec2 forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z).rotatedBy(rotation);
            camera.orientation.X.x = right.x;
            camera.orientation.X.z = right.y;
            camera.orientation.Y.x = up.x;
            camera.orientation.Y.z = up.y;
            camera.orientation.Z.x = -forward.x;
            camera.orientation.Z.z = -forward.y;
        } else if (!portal)
            resolveCollisionsWithRadius(position, movement, BODY_RADIUS);

        camera.position.x = position.x;
        camera.position.z = position.y;
        renderer.position = position;
        if (teleport)
            renderer.onScreenChanged();
    }

    void stopEditing() {
        edit_mode = Edit::None;
        render_state.hovered_pos = 0.0f;
        adding = false;
        removing = false;
        moving = false;
        dragged_enemy_index = -1;
        dragged_column_index = -1;
        added_column_index = -1;
    }

    void edit(vec2i mouse_pos) {
        if (edit_mode == Edit::None ||
            mouse_pos.x < 0 ||
            mouse_pos.y < 0 ||
            mouse_pos.x >= renderer.screen_width ||
            mouse_pos.y >= renderer.screen_height) {
            return;
        }

        const WallHitGroup& wall_hit_group{renderer.wall_hits[mouse_pos.x]};
        const GroundHit& ground_hit{renderer.ground_hits[mouse_pos.y]};

        const vec2 ground_position = renderer.position + wall_hit_group.main_hit.ray_direction * ground_hit.z;
        const f32 ground_position_distance = (ground_position - renderer.position).length();

        RayHit hit;
        hit.init();
        const vec2 direction = renderer.first_ray_direction + renderer.right_step * mouse_pos.x;
        closestHitByRayCast(hit, renderer.position, direction, true);
        int closest_enemy_id = ray_hit_enemy;
        int closest_column_id = hit.column_id == INVALID_COLUMN_ID ? -1 : hit.column_id;
        const vec2 closest_hit_position = hit.position;
        const f32 closest_hit_distance = (hit.position - renderer.position).length();

        render_state.hovered_pos = closest_hit_distance < ground_position_distance ? closest_hit_position : ground_position;

        switch (edit_mode) {
            case Edit::Walls: {
                render_state.hovered_pos = ground_position;
                Tile& tile{tile_map.cells[(i32)ground_position.y][(i32)ground_position.x]};
                bool tile_changed = false;
                if (adding) {
                    if (!tile.is_full &&
                        !((i32)ground_position.x == (i32)renderer.position.x &&
                          (i32)ground_position.y == (i32)renderer.position.y)) {
                        tile_changed = true;
                        tile.is_full = true;
                        tile.left.texture_id = tile.right.texture_id = tile.bottom.texture_id = tile.top.texture_id = 5;
                    }
                } else if (removing && tile.is_full) {
                    tile_changed = true;
                    tile.is_full = false;
                }

                if (tile_changed) {
                    TileEdge portal_from_edge;
                    TileEdge portal_to_edge;
                    if (renderer.portals.from.isActive())
                        portal_from_edge = tile_map.edges[renderer.portals.from.edge_id];
                    if (renderer.portals.to.isActive())
                        portal_to_edge = tile_map.edges[renderer.portals.to.edge_id];

                    tile_map.generateEdges();

                    if (renderer.portals.from.isActive() ||
                        renderer.portals.to.isActive()) {
                        for (u16 edge_id = 0; edge_id < tile_map.edges.size; ++edge_id) {
                            const TileEdge& edge = tile_map.edges[edge_id];
                            if (renderer.portals.from.isActive() && portal_from_edge.overlaps(edge))
                                renderer.portals.from.edge_id = edge_id;
                            if (renderer.portals.to.isActive() && portal_to_edge.overlaps(edge))
                                renderer.portals.to.edge_id = edge_id;
                        }
                    }

                    if (renderer.useGPU) uploadEdges(tile_map.edges);
                    renderer.generateWallHits();
                }
                break;
            }
            case Edit::Columns: {
                if (adding) {
                    if (tile_map.columns.size < MAX_COLUMN_COUNT) {
                        if (added_column_index == -1) {
                            resolveCollisionsWithRadius(render_state.hovered_pos, renderer.ray.direction, INITIAL_COLUMN_RADIUS);

                            f32 distance_to_body = (render_state.hovered_pos - renderer.position).length() - INITIAL_COLUMN_RADIUS - BODY_RADIUS;
                            if (distance_to_body > 0.0f) {
                                added_column_index = tile_map.columns.size++;
                                Circle& column{tile_map.columns[added_column_index]};
                                column.position = render_state.hovered_pos;
                                column.radius = INITIAL_COLUMN_RADIUS;
                                if (renderer.useGPU) uploadColumns(tile_map.columns);
                                renderer.generateWallHits();
                            }
                        }

                        if (added_column_index != -1) {
                            Circle& column{tile_map.columns[added_column_index]};
                            const vec2 movement = render_state.hovered_pos - column.position;
                            if (movement.x != 0.0f || movement.y != 0.0f)
                                resolveCollisionsWithRadius(render_state.hovered_pos, movement, column.radius, added_column_index);

                            f32 new_radius = (render_state.hovered_pos - column.position).length();
                            f32 distance_to_body = (render_state.hovered_pos - renderer.position).length() - new_radius - BODY_RADIUS;
                            if (distance_to_body <= 0.0f) new_radius += distance_to_body;
                            new_radius = fmaxf(0.1f, new_radius);
                            if (new_radius != column.radius) {
                                column.radius = new_radius;
                                if (renderer.useGPU) uploadColumns(tile_map.columns);
                                renderer.generateWallHits();
                            }
                            render_state.hovered_pos = column.position;
                        }
                    } else adding = false;
                } else if (moving) {
                    if (dragged_column_index == -1 && closest_column_id != -1) {
                        dragged_column_index = closest_column_id;
                        Circle& column{tile_map.columns[dragged_column_index]};
                        edit_offset = render_state.hovered_pos - column.position;
                        render_state.hovered_pos = column.position;
                    } else if (dragged_column_index != -1) {
                        Circle& column{tile_map.columns[dragged_column_index]};

                        displaceByRayCast(column.position, render_state.hovered_pos - edit_offset, column.radius, true);
                        resolveCollisionBetweenCircles(column.position, column.radius, {renderer.position, BODY_RADIUS});
                        render_state.hovered_pos = column.position;
                        if (renderer.useGPU) uploadColumns(tile_map.columns);
                        renderer.generateWallHits();
                    }
                } else if (removing && tile_map.columns.size > 0 && closest_column_id != -1) {
                    removing = false;
                    render_state.hovered_pos = tile_map.columns[closest_column_id].position;
                    if (tile_map.columns.size > 1)
                        tile_map.columns[closest_column_id] = tile_map.columns[tile_map.columns.size - 1];
                    tile_map.columns.size--;
                    if (renderer.useGPU) uploadColumns(tile_map.columns);
                    renderer.generateWallHits();
                }
                break;
            }
            case Edit::Enemies: {
                if (adding) {
                    if (render_state.enemy_count < MAX_ENEMIES) {
                        resolveCollisionsWithRadius(render_state.hovered_pos, renderer.ray.direction, 0.5f);

                        f32 distance_to_body = (render_state.hovered_pos - renderer.position).length() - 0.5f - BODY_RADIUS;
                        if (distance_to_body > 0.0f) {
                            dragged_enemy_index = render_state.enemy_count++;
                            enemies[dragged_enemy_index].position = vec3{render_state.hovered_pos.x, 0.0f, render_state.hovered_pos.y};
                            edit_offset = 0.0f;
                            adding = false;
                            moving = true;
                        }
                    }

                     // =

                    // bool can_add = render_state.enemy_count < MAX_ENEMIES && !(
                        // position.x <= 1.25f ||
                        // position.x >= ((f32)tile_map.width - 1.75f) ||
                        // position.y <= 1.25f ||
                        // position.y >= ((f32)tile_map.height - 1.75f) ||
                        // tile_map.cells[(i32)position.y][(i32)position.x].is_full ||
                        // tile_map.cells[(i32)(position.y + 0.25f)][(i32)position.x].is_full ||
                        // tile_map.cells[(i32)(position.y - 0.25f)][(i32)position.x].is_full ||
                        // tile_map.cells[(i32)position.y][(i32)(position.x + 0.25f)].is_full ||
                        // tile_map.cells[(i32)position.y][(i32)(position.x - 0.25f)].is_full ||
                        // tile_map.cells[(i32)(position.y + 0.25f)][(i32)(position.x + 0.25f)].is_full ||
                        // tile_map.cells[(i32)(position.y - 0.25f)][(i32)(position.x + 0.25f)].is_full ||
                        // tile_map.cells[(i32)(position.y + 0.25f)][(i32)(position.x - 0.25f)].is_full ||
                        // tile_map.cells[(i32)(position.y - 0.25f)][(i32)(position.x - 0.25f)].is_full);

                    // for (u32 i = 0; i < tile_map.columns.size; i++)
                    //     if ((tile_map.columns.data[i].position - position).length() < tile_map.columns.data[i].radius)
                    //         return false;

                    // if (render_state.enemy_count < MAX_ENEMIES && enemyCanBePlaced()) {
                    //     dragged_enemy_index = render_state.enemy_count++;
                    //     enemies[dragged_enemy_index].position = vec3{position.x, 0.0f, position.y};
                    //     adding = false;
                    //     moving = true;
                    // }
                } else if (dragged_enemy_index == -1 && closest_enemy_id != -1) {
                    dragged_enemy_index = closest_enemy_id;
                    vec2 enemy_position{enemies[closest_enemy_id].position.x, enemies[closest_enemy_id].position.z};
                    if (moving)
                        edit_offset = render_state.hovered_pos - enemy_position;
                    render_state.hovered_pos = enemy_position;
                }

                if (dragged_enemy_index != -1) {
                    if (moving) {
                        vec2 enemy_position{enemies[dragged_enemy_index].position.x, enemies[dragged_enemy_index].position.z};
                        displaceByRayCast(enemy_position, render_state.hovered_pos - edit_offset, 0.5f, true);
                        resolveCollisionBetweenCircles(enemy_position, 0.5f, {renderer.position, BODY_RADIUS});
                        enemies[dragged_enemy_index].position = vec3{enemy_position.x, 0.0f, enemy_position.y};
                        render_state.hovered_pos = enemy_position;
                    } else if (removing) {
                        removing = false;
                        if (render_state.enemy_count) {
                            if (render_state.enemy_count > 1) {
                                enemies[dragged_enemy_index] = enemies[render_state.enemy_count - 1];
                                render_state.enemies[dragged_enemy_index] = render_state.enemies[render_state.enemy_count - 1];
                            }
                            render_state.enemy_count--;
                        }
                    }
                }

                break;
            }
        }
    }
};
