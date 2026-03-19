#pragma once

#include "../scene/camera.h"
#include "../scene/tilemap.h"
#include "ray_caster.h"
#include "pixel_shader.h"

#ifdef __CUDACC__
#include "./renderer_GPU.h"
#else
#define USE_GPU_BY_DEFAULT false
void initDataOnGPU(const RayCasterSettings& settings) {}
void uploadEdges(const Slice<TileEdge>& edges) {}
void uploadColumns(const Slice<Circle>& columns) {}
void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {}
void generateWallHitsOnGPU(const RayCaster &ray_caster) {}
void uploadWallHits(WallHitGroup* wall_hit_groups, u16 wall_hits_count)  {}
void downloadWallHits(WallHitGroup* wall_hits, u16 wall_hits_count)  {}
#endif



#define INVALID_PROJECTILE_INDEX ((u8)(-1))

namespace ray_cast_renderer {
    RayCasterSettings* settings;
    RayCaster ray_caster;
    f32 prior_up_aim;
    u16 prior_screen_height;
    bool useGPU = false;
    bool adding_column = false;
    bool adding_tiles = false;
    bool removing_tiles = false;

    RenderState render_state;

    WallHitGroup wall_hits[MAX_WALL_HITS_COUNT];
    GroundHit ground_hits[MAX_GROUND_HITS_COUNT];

    SpinningProjectile projectiles[MAX_POINT_LIGHTS];
    u8 projectile_count = 0;

    Color torch_light_color{1.0f, 0.6f, 0.35f};
    f32 torch_light_intensity = 4.0f;

    struct PortalState {
        Color color;
        f32 spawned_time;
        u16 projectile_index;

        void update(Portal& portal, const f32 time, const TileEdge* edges) {
            const f32 elapsed_time = time - spawned_time;
            if (elapsed_time <= PORTAL_GROW_TIME) {
                portal.radius = PORTAL_GROW_RADIUS +
                                PORTAL_GROW_RANGE * smoothStep(0.0f, 1.0f, PORTAL_GROW_RATE * elapsed_time);
                if ((fabsf(portal.position.y) + (portal.radius * 2.0f) + PORTAL_BREATHING_RANGE) > 1.0f)
                    portal.position.y = (1.0f - (portal.radius * 2.0f) - PORTAL_BREATHING_RANGE) * (portal.position.y > 0.0f ? 1.0f : -1.0f);

                const TileEdge& edge{edges[portal.edge_id]};
                if (edge.is & (FACING_DOWN | FACING_UP)) {
                    if ((portal.position.x + portal.radius + PORTAL_BREATHING_RANGE) > edge.to.x)
                        portal.position.x = edge.to.x - portal.radius - PORTAL_BREATHING_RANGE;
                    else if ((portal.position.x - portal.radius - PORTAL_BREATHING_RANGE) < edge.from.x)
                        portal.position.x = edge.from.x + portal.radius + PORTAL_BREATHING_RANGE;
                } else {
                    if ((portal.position.z + portal.radius + PORTAL_BREATHING_RANGE) > edge.to.y)
                        portal.position.z = edge.to.y - portal.radius - PORTAL_BREATHING_RANGE;
                    else if ((portal.position.z - portal.radius - PORTAL_BREATHING_RANGE) < edge.from.y)
                        portal.position.z = edge.from.y + portal.radius + PORTAL_BREATHING_RANGE;
                }
            } else
                portal.radius = PORTAL_FINAL_RADIUS + PORTAL_BREATHING_RANGE * cos((elapsed_time - PORTAL_GROW_TIME) * 2.0f);
        }
    };

    PortalState portal_from{Cyan, 0.0f, INVALID_PROJECTILE_INDEX};
    PortalState portal_to{Magenta, 0.0f, INVALID_PROJECTILE_INDEX};

    void toggleUseOfGPU(const TileMap& tile_map) {
#ifdef __CUDACC__
        if (useGPU) {
            downloadWallHits(wall_hits, ray_caster.screen_width);
            useGPU = false;
        } else {
            uploadWallHits(wall_hits, ray_caster.screen_width);
            uploadColumns(tile_map.columns);
            uploadEdges(tile_map.edges);
            useGPU = true;
        }
#endif
    }

    void generateFloorAndCeilingHits() {
        f32 Y = 1.0f + ray_caster.up_aim;

        f32 screen_pixel_height = 2.0f / (f32)ray_caster.screen_height;

        f32 Z, priorZ = 1.0f / (Y + screen_pixel_height);
        i32 y = 0;

        for (; y < ray_caster.mid_point; y++, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z * 2.0f;
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        Y = 1.0f - ray_caster.up_aim;
        priorZ = 1.0f / (Y + screen_pixel_height);
        y = ray_caster.screen_height - 1;

        for (; y > ray_caster.mid_point; y--, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z * 2.0f;
            ground_hits[y].mip = computeMip(Z - priorZ, ray_caster.texel_size, ray_caster.last_mip);
            priorZ = Z;
        }

        uploadGroundHits(ground_hits, ray_caster.screen_height);
    }

    void generateWallHits(const TileMap& tile_map) {
        if (useGPU) {
            generateWallHitsOnGPU(ray_caster);
            downloadWallHits(wall_hits, ray_caster.screen_width);
        } else {
            WallHitGroup wall_hit_group;
            vec2 ray_direction = ray_caster.first_ray_direction;
            for (u16 x = 0; x < ray_caster.screen_width; x++, ray_direction += ray_caster.right_step) {
                ray_caster.generateWallHitWithPortals(wall_hit_group, ray_direction, tile_map.edges, tile_map.columns);
                wall_hits[x] = wall_hit_group;
            }
        }
    }

    void addLightProjectile(const f32 time, const Color color) {
        SpinningProjectile& projectile{projectiles[projectile_count++]};
        PointLight& point_light{render_state.lights[render_state.light_count++]};

        projectile.init(ray_caster.position, ray_caster.forward, ray_caster.up_aim, settings->projectile_radius, time);

        point_light.position = projectile.position;
        point_light.color = color;
        point_light.intensity = torch_light_intensity * 0.25f;
    }

    void fireFlare(const f32 time) {
        if (render_state.light_count < (MAX_POINT_LIGHTS - 2))
            addLightProjectile(time, torch_light_color);
    }

    void launchPortalFrom(const f32 time) {
        portal_from.projectile_index = projectile_count;
        addLightProjectile(time, portal_from.color);
    }

    void launchPortalTo(const f32 time) {
        portal_to.projectile_index = projectile_count;
        addLightProjectile(time, portal_to.color);
    }

    void update(const f32 time, const f32 delta_time, const TileMap& tile_map) {
        if (ray_caster.portal_from.edge_id != INVALID_EDGE_ID)
            portal_from.update(ray_caster.portal_from, time, tile_map.edges.data);

        if (ray_caster.portal_to.edge_id != INVALID_EDGE_ID)
            portal_to.update(ray_caster.portal_to, time, tile_map.edges.data);

        if (projectile_count == 0)
            return;

        bool need_generate_wall_hits = false;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(settings->tile_map_width - 1),
            (f32)(settings->tile_map_height - 1)
        };
        for (u16 i = 0; i < projectile_count; i++) {
            SpinningProjectile& projectile{projectiles[i]};
            const f32 elapsed_time = time - projectile.spawned_time;
            vec3 projectile_position = projectile.position;
            projectile.updatePosition(delta_time * settings->projectile_speed);

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
                ray.update(vec2{projectile_position.x, projectile_position.z}, ray_direction_2d / distance_2d, ray_caster.forward);
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

                if (i == portal_from.projectile_index ||
                    i == portal_to.projectile_index) {
                    const bool is_from = portal_from.projectile_index == i;

                    Portal& portal{      is_from ? ray_caster.portal_from : ray_caster.portal_to};
                    Portal& other_portal{is_from ? ray_caster.portal_to : ray_caster.portal_from};

                    if (other_portal.edge_id == INVALID_EDGE_ID ||
                        (other_portal.position - new_projectile_position).length() > (2 * PORTAL_FINAL_RADIUS)) {
                        PortalState& portal_state{is_from ? portal_from : portal_to};
                        portal_state.spawned_time = time;
                        portal_state.projectile_index = INVALID_PROJECTILE_INDEX;

                        portal.position = new_projectile_position;
                        portal.edge_id = closest_hit_edge_id;
                        portal.edge_is = closest_hit_edge_is;
                        portal.radius = PORTAL_INITIAL_RADIUS;
                        portal.color = portal_state.color;

                        if (other_portal.edge_id != INVALID_EDGE_ID)
                            need_generate_wall_hits = true;
                    }
                } else if (closest_hit_edge_id != INVALID_EDGE_ID) {
                    Portal* portal = nullptr;
                    if (ray_caster.portal_from.edge_id == closest_hit_edge_id) {
                        portal = &ray_caster.portal_from;
                        const vec3 PortalToP =
                            vec3{new_projectile_position.x, new_projectile_position.y * 0.5f, new_projectile_position.z} -
                            vec3{portal->position.x, portal->position.y * 0.5f, portal->position.z};
                        if (PortalToP.squaredLength() > (portal->radius * portal->radius))
                            portal = nullptr;
                    }
                    if (portal == nullptr && ray_caster.portal_to.edge_id == closest_hit_edge_id) {
                        portal = &ray_caster.portal_to;
                        const vec3 PortalToP =
                            vec3{new_projectile_position.x, new_projectile_position.y * 0.5f, new_projectile_position.z} -
                                vec3{portal->position.x, portal->position.y * 0.5f, portal->position.z};
                        if (PortalToP.squaredLength() > (portal->radius * portal->radius))
                            portal = nullptr;
                    }

                    if (portal) {
                        Portal* other_portal{portal == &ray_caster.portal_from ? &ray_caster.portal_to : &ray_caster.portal_from};
                        if (other_portal->edge_id != INVALID_EDGE_ID) {
                            i32 ray_rotation = portal->getRotation(other_portal->edge_is);

                            vec2 origin{projectile_position.x, projectile_position.z};
                            vec2 origin_to_hit_position = {projectile_to_edge.x, projectile_to_edge.z};
                            vec2 origin_to_portal = vec2{portal->position.x, portal->position.z} - origin;
                            vec2 forward{projectile.forward.x, projectile.forward.z};
                            if (ray_rotation == 90) {
                                origin_to_hit_position = origin_to_hit_position.ccw90();
                                origin_to_portal = origin_to_portal.ccw90();
                                ray_direction_2d = ray_direction_2d.ccw90();
                                forward = forward.ccw90();
                            } else if (ray_rotation == -90) {
                                origin_to_hit_position = origin_to_hit_position.cw90();
                                origin_to_portal = origin_to_portal.cw90();
                                ray_direction_2d = ray_direction_2d.cw90();
                                forward = forward.cw90();
                            } else if (ray_rotation == 180) {
                                origin_to_hit_position = -origin_to_hit_position;
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
                    if (portal_from.projectile_index == 0)
                        portal_from.projectile_index = INVALID_EDGE_ID;
                    if (portal_to.projectile_index == 0)
                        portal_to.projectile_index = INVALID_EDGE_ID;

                    if (need_generate_wall_hits)
                        generateWallHits(tile_map);

                    return;
                }

                if (portal_from.projectile_index == projectile_count)
                    portal_from.projectile_index = i;
                if (portal_to.projectile_index == projectile_count)
                    portal_to.projectile_index = i;

                projectiles[i] = projectiles[projectile_count];
                render_state.lights[i] = render_state.lights[render_state.light_count];
                i--;
                continue;
            }

            PointLight& point_light{render_state.lights[i+1]};
            point_light.position = projectile.position;
            if (i == portal_from.projectile_index)
                point_light.flicker(portal_from.color, torch_light_intensity * 0.25f, elapsed_time);
            else if (i == portal_to.projectile_index)
                point_light.flicker(portal_to.color, torch_light_intensity * 0.25f, elapsed_time);
            else
                point_light.flicker(torch_light_color, torch_light_intensity * 0.25f, elapsed_time);
        }

        if (need_generate_wall_hits)
            generateWallHits(tile_map);
    }

    void onMove(Camera& camera, TileMap& tile_map) {
        vec2 position = vec2(camera.position.x, camera.position.z);
        vec2 movement = position - ray_caster.position;

        if (movement.x > 0.0f) {
            i32 next_pos = (i32)(position.x + settings->body_radius);
            const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
            if (next_tile.is_full) position.x = (f32)next_pos - settings->body_radius;
        } else if (movement.x < 0.0f) {
            i32 next_pos = (i32)(position.x - settings->body_radius);
            const Tile& next_tile = tile_map.cells[(i32)position.y][next_pos];
            if (next_tile.is_full) position.x = (f32)(next_pos + 1) + settings->body_radius;
        }

        if (movement.y < 0.0f) {
            i32 next_pos = (i32)(position.y - settings->body_radius);
            const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
            if (next_tile.is_full) position.y = (f32)(next_pos + 1) + settings->body_radius;
        } else if (movement.y > 0.0f) {
            i32 next_pos = (i32)(position.y + settings->body_radius);
            const Tile& next_tile = tile_map.cells[next_pos][(i32)position.x];
            if (next_tile.is_full) position.y = (f32)next_pos - settings->body_radius;
        }

        for (u32 i = 0; i < tile_map.columns.size; i++) {
            const Circle& column{tile_map.columns.data[i]};

            vec2 vector_to_column = column.position - position;
            f32 distance_to_column = vector_to_column.length();
            f32 min_distance_allowed = settings->body_radius + column.radius;
            if (distance_to_column < min_distance_allowed)
                position -= (vector_to_column / distance_to_column) * (min_distance_allowed - distance_to_column);
        }

        camera.position.x = position.x;
        camera.position.z = position.y;
        ray_caster.position = position;
    }

    void onScreenChanged(const Camera& camera, const TileMap& tile_map) {
        vec2 right = vec2(camera.orientation.X.x, camera.orientation.X.z);
        vec2 forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z);
        ray_caster.onScreenChanged(camera.focal_length, forward, right, camera.orientation.Z.y);
        generateWallHits(tile_map);
        if (prior_screen_height != ray_caster.screen_height ||
            prior_up_aim != ray_caster.up_aim)
            generateFloorAndCeilingHits();

        prior_up_aim = ray_caster.up_aim;
    }

    void onResize(u16 width, u16 height, const Camera& camera, const TileMap& tile_map) {
        ray_caster.screen_height = (height >> 1) << 1;
        ray_caster.screen_width = width;
        onScreenChanged(camera, tile_map);

        prior_screen_height = ray_caster.screen_height;
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
            mouse_pos.x >= ray_caster.screen_width ||
            mouse_pos.y >= ray_caster.screen_height) {
            return;
        }

        const WallHitGroup& wall_hit_group{wall_hits[mouse_pos.x]};
        const GroundHit& ground_hit{ground_hits[mouse_pos.y]};

        const vec2 position = ray_caster.position + wall_hit_group.main.ray_direction * ground_hit.z;
        const vec2 start = 1.0f;
        const vec2 end = {
            (f32)(settings->tile_map_width - 1),
            (f32)(settings->tile_map_height - 1)
        };
        if (!inRange(start, position, end)) {
            return;
        }

        render_state.hovered_pos = position;

        if (crete_new_column) {
            f32 distance_to_body = (position - ray_caster.position).length() - settings->initial_column_radius - settings->body_radius;
            if (distance_to_body > 0.0f) {
                Circle& column{tile_map.columns[tile_map.columns.size++]};
                column.position = position;
                column.radius = settings->initial_column_radius;
                if (useGPU) uploadColumns(tile_map.columns);
                generateWallHits(tile_map);
                render_state.hovered_pos = 0.0f;
            }
        } else if (adding_column) {
            Circle& column{tile_map.columns[tile_map.columns.size - 1]};
            f32 new_radius = (position - column.position).length();
            f32 distance_to_body = (position - ray_caster.position).length() - new_radius - settings->body_radius;
            if (distance_to_body <= 0.0f) new_radius += distance_to_body;
            new_radius = fmaxf(0.1f, new_radius);
            if (new_radius != column.radius) {
                column.radius = new_radius;
                if (useGPU) uploadColumns(tile_map.columns);
                generateWallHits(tile_map);
            }
            render_state.hovered_pos = 0.0f;
        } else {
            Tile& tile{tile_map.cells[(i32)position.y][(i32)position.x]};
            bool tile_changed = false;
            if (adding_tiles) {
                if (!tile.is_full &&
                    !((i32)position.x == (i32)ray_caster.position.x &&
                      (i32)position.y == (i32)ray_caster.position.y)) {
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
                if (ray_caster.portal_from.edge_id != INVALID_EDGE_ID)
                    portal_from_edge = tile_map.edges[ray_caster.portal_from.edge_id];
                if (ray_caster.portal_to.edge_id != INVALID_EDGE_ID)
                    portal_to_edge = tile_map.edges[ray_caster.portal_to.edge_id];

                generateTileMapEdges(tile_map);

                if (ray_caster.portal_from.edge_id != INVALID_EDGE_ID ||
                    ray_caster.portal_to.edge_id != INVALID_EDGE_ID) {
                    for (u16 edge_id = 0; edge_id < tile_map.edges.size; ++edge_id) {
                        const TileEdge& edge = tile_map.edges[edge_id];
                        if (ray_caster.portal_from.edge_id != INVALID_EDGE_ID &&
                            portal_from_edge.overlaps(edge))
                            ray_caster.portal_from.edge_id = edge_id;
                        if (ray_caster.portal_to.edge_id != INVALID_EDGE_ID &&
                            portal_to_edge.overlaps(edge))
                            ray_caster.portal_to.edge_id = edge_id;
                    }
                }

                if (useGPU) uploadEdges(tile_map.edges);
                generateWallHits(tile_map);
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
            const WallHit& wall_hit{wall_hits[mouse_pos.x].main};
            if (wall_hit.column_id != INVALID_COLUMN_ID) {
                tile_map.columns[wall_hit.column_id] = tile_map.columns[--tile_map.columns.size];
                if (useGPU) uploadColumns(tile_map.columns);
                generateWallHits(tile_map);
            }
        }
    }

    void renderOnCPU(u32* window_content, const TileMap& tile_map) {
        PixelShader pixel_shader{*settings, render_state};
        u32 offset = 0;
        for (u16 y = 0; y < ray_caster.screen_height; y++) {
            GroundHit ground_hit = ground_hits[y];
            for (u16 x = 0; x < ray_caster.screen_width; x++, offset++) {
                const WallHitGroup& wall_hit_group{wall_hits[x]};
                window_content[offset] = pixel_shader.shade(
                    ground_hit,
                    wall_hit_group.main,
                    ray_caster.portal_from,
                    ray_caster.portal_to,
                    tile_map.edges,
                    tile_map.columns,
                    ray_caster.position,
                    y,
                    ray_caster.mid_point,
                    wall_hit_group.portal,
                    wall_hit_group.portal_origin).asContent();
            }
        }
    }

    void init(RayCasterSettings* render_settings, const Dimensions& dim, Camera& camera, TileMap& tile_map)
    {
        settings = render_settings;
        render_state.init();

        ray_caster.portal_from.init();
        ray_caster.portal_to.init();

        Texture &texture{settings->textures[0]};
        ray_caster.texel_size = 1.0f / (f32)texture.width;
        ray_caster.last_mip = (u8)(texture.mip_count - 1);

        initDataOnGPU(*settings);
        uploadEdges(tile_map.edges);

        prior_screen_height = 0;
        prior_up_aim = 0.0f;

        onMove(camera, tile_map);
        onResize(dim.width, dim.height, camera, tile_map);
    }

    void render(u32* window_content, const TileMap& tile_map) {
        #ifdef __CUDACC__
        if (useGPU) renderOnGPU(ray_caster, render_state, window_content);
        else        renderOnCPU(window_content, tile_map);
        #else
        renderOnCPU(window_content, tile_map);
        #endif
    }
};