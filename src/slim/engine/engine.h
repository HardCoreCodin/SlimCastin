#pragma once

#include "renderer.h"

#include "../scene/camera.h"


struct Engine {
    RenderData &render_data;
    RenderState &render_state;

    Renderer renderer;

    Engine() :
        render_data{renderer.render_data},
        render_state{renderer.render_state}
    {}

    void init(TileMap& tile_map, const RenderData &map_render_data, const Dimensions& dim, Camera& camera) {
        renderer.init(map_render_data);
        uploadEdges(tile_map.edges);
        onMove(camera, tile_map);
        onResize(dim.width, dim.height, camera, tile_map);
    }

    void render(u32* window_content, const TileMap& tile_map) {
        renderer.render(window_content, tile_map);
    }

    struct Projectile {
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
    Projectile projectiles[MAX_POINT_LIGHTS];
    u8 projectile_count = 0;

    Color torch_light_color{1.0f, 0.6f, 0.35f};
    f32 torch_light_intensity = 4.0f;

    bool adding_column = false;
    bool adding_tiles = false;
    bool removing_tiles = false;

    void addLightProjectile(const f32 time, const Color color) {
        Projectile& projectile{projectiles[projectile_count++]};
        PointLight& point_light{render_state.lights[render_state.light_count++]};

        projectile.init(renderer.position, renderer.forward, renderer.up_aim, time);

        point_light.position = projectile.position;
        point_light.color = color;
        point_light.intensity = torch_light_intensity * 0.25f;
    }

    void fireFlare(const f32 time) {
        if (render_state.light_count < (MAX_POINT_LIGHTS - 2))
            addLightProjectile(time, torch_light_color);
    }

    void launchPortalFrom(const f32 time) {
        renderer.portals.from.projectile_index = projectile_count;
        addLightProjectile(time, renderer.portals.from.color);
    }

    void launchPortalTo(const f32 time) {
        renderer.portals.to.projectile_index = projectile_count;
        addLightProjectile(time, renderer.portals.to.color);
    }

    void updateProjectiles(const f32 time, const f32 delta_time, const TileMap& tile_map) {
        bool need_generate_wall_hits = false;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(render_data.map_width - 1),
            (f32)(render_data.map_height - 1)
        };
        for (u8 i = 0; i < projectile_count; i++) {
            Projectile& projectile{projectiles[i]};
            const f32 elapsed_time = time - projectile.spawned_time;
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
                const f32 distance_2d = ray_direction_2d.length();
                ray.update(vec2{projectile_position.x, projectile_position.z}, ray_direction_2d / distance_2d, renderer.forward);
                f32 hit_distance = 1000000.0f;
                u16 closest_hit_edge_id = INVALID_EDGE_ID;
                u8 closest_hit_edge_is = 0;
                for (u16 edge_id = 0; edge_id < (u16)tile_map.edges.size; edge_id++) {
                    if (ray.intersectsWithEdge(tile_map.edges.data[edge_id])) {
                        ray.hit.distance = (ray.hit.position - ray.origin).squaredLength();
                        if (ray.hit.distance < hit_distance) {
                            hit_distance = ray.hit.distance;
                            closest_hit_edge_id = edge_id;
                            closest_hit_edge_is = ray.hit.edge_is;
                        }
                    }
                }

                const vec3 projectile_to_edge = ray_direction_3d * (sqrt(hit_distance) / distance_2d);
                const vec3 new_projectile_position = projectile_position + projectile_to_edge;

                const Portal* other_portal = nullptr;
                if (closest_hit_edge_id != INVALID_EDGE_ID) {
                    const Portal* portal = renderer.portals.getPortalsFromWallPosition3D(new_projectile_position, closest_hit_edge_id, &other_portal);

                    if (portal && other_portal->isActive()) {
                        i32 ray_rotation = portal->getRotation(other_portal->edge_is);

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
                }
                if (!teleported) {
                    Portal* new_portal = renderer.portals.getPortalsFromProjectileIndex(i, &other_portal);
                    // Check if spawning a portal at the projectile's hit position is allowed:
                    if (new_portal && (
                        !other_portal->isActive() ||
                        (other_portal->position - new_projectile_position).length() > (2 * PORTAL_FINAL_RADIUS))) {
                        new_portal->spawned_time = time;
                        new_portal->projectile_index = INVALID_PROJECTILE_INDEX;
                        new_portal->position = new_projectile_position;
                        new_portal->edge_id = closest_hit_edge_id;
                        new_portal->edge_is = closest_hit_edge_is;
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
                        renderer.generateWallHits(tile_map);

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

            PointLight& point_light{render_state.lights[i+1]};
            point_light.position = projectile.position;
            if (i == renderer.portals.from.projectile_index)
                point_light.flicker(renderer.portals.from.color, torch_light_intensity * 0.25f, elapsed_time);
            else if (i == renderer.portals.to.projectile_index)
                point_light.flicker(renderer.portals.to.color, torch_light_intensity * 0.25f, elapsed_time);
            else
                point_light.flicker(torch_light_color, torch_light_intensity * 0.25f, elapsed_time);
        }

        if (need_generate_wall_hits)
            renderer.generateWallHits(tile_map);
    }

    void update(const f32 time, const f32 delta_time, const TileMap& tile_map) {
        renderer.portals.update(time, tile_map.edges.data);

        if (projectile_count > 0)
            updateProjectiles(time, delta_time, tile_map);

        if (renderer.portals.areBothActive())
            renderer.updatePortalLights();
    }

    void onScreenChanged(const Camera& camera, const TileMap& tile_map) {
        vec2 right = vec2(camera.orientation.X.x, camera.orientation.X.z).normalized();
        vec2 forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z).normalized();
        renderer.onScreenChanged(tile_map, camera.focal_length, forward, right, camera.orientation.Z.y);
    }

    void onMove(Camera& camera, TileMap& tile_map) {
        vec2 position = vec2(camera.position.x, camera.position.z);
        vec2 movement = position - renderer.position;

        bool teleport = false;
        const Portal* other_portal = nullptr;
        const Portal* portal = nullptr;
        if (renderer.portals.areBothActive()) {
            Ray ray;

            ray.update(renderer.position, movement, renderer.forward);
            f32 hit_distance = 1000000.0f;
            u16 closest_hit_edge_id = INVALID_EDGE_ID;
            u8 closest_hit_edge_is = 0;
            for (u16 edge_id = 0; edge_id < (u16)tile_map.edges.size; edge_id++) {
                if (ray.intersectsWithEdge(tile_map.edges.data[edge_id])) {
                    ray.hit.distance = (ray.hit.position - ray.origin).squaredLength();
                    if (ray.hit.distance < hit_distance) {
                        hit_distance = ray.hit.distance;
                        closest_hit_edge_id = edge_id;
                        closest_hit_edge_is = ray.hit.edge_is;
                    }
                }
            }

            portal = renderer.portals.getPortalsFromWallPosition2D(position, closest_hit_edge_id, &other_portal);
            if (portal) {
                const vec2 new_position = position + movement;
                if (closest_hit_edge_is & (FACING_UP | FACING_DOWN))
                    teleport = new_position.y > portal->position.z != position.y > portal->position.z;
                else
                    teleport = new_position.x > portal->position.x != position.x > portal->position.x;
            }
        }

        if (teleport) {
            i32 ray_rotation = portal->getRotation(other_portal->edge_is);

            vec2 origin_to_portal = vec2{portal->position.x, portal->position.z} - renderer.position;

            vec2 up = vec2(camera.orientation.Y.x, camera.orientation.Y.z);
            vec2 right = vec2(camera.orientation.X.x, camera.orientation.X.z);
            vec2 forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z);
            if (ray_rotation == 90) {
                origin_to_portal = origin_to_portal.ccw90();
                movement = movement.ccw90();
                up = up.ccw90();
                right = right.ccw90();
                forward = forward.ccw90();
                camera.orientation.y += DEG_TO_RAD * 90;
            } else if (ray_rotation == -90) {
                origin_to_portal = origin_to_portal.cw90();
                movement = movement.cw90();
                up = up.cw90();
                right = right.cw90();
                forward = forward.cw90();
                camera.orientation.y -= DEG_TO_RAD * 90;
            } else if (ray_rotation == 180) {
                origin_to_portal = -origin_to_portal;
                movement = -movement;
                up = -up;
                right = -right;
                forward = -forward;
                camera.orientation.y += DEG_TO_RAD * 180;
            }
            const vec2 other_portal_origin = vec2{other_portal->position.x, other_portal->position.z} - origin_to_portal;
            position = other_portal_origin + movement;

            camera.orientation.X.x = right.x;
            camera.orientation.X.z = right.y;
            camera.orientation.Y.x = up.x;
            camera.orientation.Y.z = up.y;
            camera.orientation.Z.x = -forward.x;
            camera.orientation.Z.z = -forward.y;
        } else if (!portal) {
            if (movement.x > 0.0f) {
                i32 next_pos = (i32)(position.x + BODY_RADIUS);
                const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
                if (next_tile.is_full) position.x = (f32)next_pos - BODY_RADIUS;
            } else if (movement.x < 0.0f) {
                i32 next_pos = (i32)(position.x - BODY_RADIUS);
                const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
                if (next_tile.is_full) position.x = (f32)(next_pos + 1) + BODY_RADIUS;
            }

            if (movement.y < 0.0f) {
                i32 next_pos = (i32)(position.y - BODY_RADIUS);
                const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
                if (next_tile.is_full) position.y = (f32)(next_pos + 1) + BODY_RADIUS;
            } else if (movement.y > 0.0f) {
                i32 next_pos = (i32)(position.y + BODY_RADIUS);
                const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
                if (next_tile.is_full) position.y = (f32)next_pos - BODY_RADIUS;
            }

            for (u32 i = 0; i < tile_map.columns.size; i++) {
                const Circle& column{tile_map.columns.data[i]};

                vec2 vector_to_column = column.position - position;
                f32 distance_to_column = vector_to_column.length();
                f32 min_distance_allowed = BODY_RADIUS + column.radius;
                if (distance_to_column < min_distance_allowed)
                    position -= (vector_to_column / distance_to_column) * (min_distance_allowed - distance_to_column);
            }
        }

        camera.position.x = position.x;
        camera.position.z = position.y;
        renderer.position = position;
        if (teleport)
            onScreenChanged(camera, tile_map);
    }

    void onResize(u16 width, u16 height, const Camera& camera, const TileMap& tile_map) {
        renderer.screen_height = (height >> 1) << 1;
        renderer.screen_width = width;
        onScreenChanged(camera, tile_map);

        renderer.prior_screen_height = renderer.screen_height;
    }

    void onStopEditing() {
        render_state.hovered_pos = 0.0f;
        adding_column = false;
        adding_tiles = false;
        removing_tiles = false;
    }

    void onEditHover(TileMap& tile_map, vec2i mouse_pos, bool crete_new_column = false) {
        if ((render_state.flags & (EDITING_WALLS | EDITING_COLUMNS)) == 0 ||
            mouse_pos.x < 0 ||
            mouse_pos.y < 0 ||
            mouse_pos.x >= renderer.screen_width ||
            mouse_pos.y >= renderer.screen_height) {
            return;
        }

        const WallHitGroup& wall_hit_group{renderer.wall_hits[mouse_pos.x]};
        const GroundHit& ground_hit{renderer.ground_hits[mouse_pos.y]};

        const vec2 position = renderer.position + wall_hit_group.main.ray_direction * ground_hit.z;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(render_data.map_width - 1),
            (f32)(render_data.map_height - 1)
        };
        if (!inRange(start, position, end)) {
            return;
        }

        render_state.hovered_pos = position;

        if (crete_new_column) {
            f32 distance_to_body = (position - renderer.position).length() - INITIAL_COLUMN_RADIUS - BODY_RADIUS;
            if (distance_to_body > 0.0f) {
                Circle& column{tile_map.columns[tile_map.columns.size++]};
                column.position = position;
                column.radius = INITIAL_COLUMN_RADIUS;
                if (renderer.useGPU) uploadColumns(tile_map.columns);
                renderer.generateWallHits(tile_map);
                render_state.hovered_pos = 0.0f;
            }
        } else if (adding_column) {
            Circle& column{tile_map.columns[tile_map.columns.size - 1]};
            f32 new_radius = (position - column.position).length();
            f32 distance_to_body = (position - renderer.position).length() - new_radius - BODY_RADIUS;
            if (distance_to_body <= 0.0f) new_radius += distance_to_body;
            new_radius = fmaxf(0.1f, new_radius);
            if (new_radius != column.radius) {
                column.radius = new_radius;
                if (renderer.useGPU) uploadColumns(tile_map.columns);
                renderer.generateWallHits(tile_map);
            }
            render_state.hovered_pos = 0.0f;
        } else {
            Tile& tile{tile_map.cells[(i32)position.y][(i32)position.x]};
            bool tile_changed = false;
            if (adding_tiles) {
                if (!tile.is_full &&
                    !((i32)position.x == (i32)renderer.position.x &&
                      (i32)position.y == (i32)renderer.position.y)) {
                    tile_changed = true;
                    tile.is_full = true;
                    tile.left.texture_id = tile.right.texture_id = tile.bottom.texture_id = tile.top.texture_id = 12;
                }
            } else if (removing_tiles && tile.is_full) {
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
                renderer.generateWallHits(tile_map);
            }
        }
    }

    void onEditLeftMouseButtonDown(TileMap& tile_map, vec2i mouse_pos) {
        if (render_state.flags & EDITING_WALLS) {
            onStopEditing();
            adding_tiles = true;
            onEditHover(tile_map, mouse_pos);
        } else if ((render_state.flags & EDITING_COLUMNS) && (tile_map.columns.size < MAX_COLUMN_COUNT)) {
            adding_column = true;
            onEditHover(tile_map, mouse_pos, true);
        }
    }

    void onEditRightMouseButtonDown(TileMap& tile_map, vec2i mouse_pos) {
        if (render_state.flags & EDITING_WALLS) {
            onStopEditing();
            removing_tiles = true;
            onEditHover(tile_map, mouse_pos);
        } else if ((render_state.flags & EDITING_COLUMNS) && (tile_map.columns.size != 0)) {
            const WallHit& wall_hit{renderer.wall_hits[mouse_pos.x].main};
            if (wall_hit.column_id != INVALID_COLUMN_ID) {
                tile_map.columns[wall_hit.column_id] = tile_map.columns[--tile_map.columns.size];
                if (renderer.useGPU) uploadColumns(tile_map.columns);
                renderer.generateWallHits(tile_map);
            }
        }
    }
};
