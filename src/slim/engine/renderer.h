#pragma once

#include "tilemap.h"
#include "raycast.h"
#include "pixel_shader.h"

#ifdef __CUDACC__
#include "./renderer_GPU.h"
#else
#define USE_GPU_BY_DEFAULT false
void initDataOnGPU(const RenderData& render_data) {}
void uploadEdges(const Slice<TileEdge>& edges) {}
void uploadColumns(const Slice<Circle>& columns) {}
void uploadGroundHits(GroundHit* ground_hits, u16 ground_hits_count) {}
void generateWallHitsOnGPU(const RayCast &raycast) {}
void uploadWallHits(WallHitGroup* wall_hit_groups, u16 wall_hits_count)  {}
void downloadWallHits(WallHitGroup* wall_hits, u16 wall_hits_count)  {}
#endif


struct Renderer : RayCast {
    Camera& camera;
    Slice<Circle> &columns;
    Slice<TileEdge> &edges;

    Renderer(Camera& camera, Slice<Circle> &columns, Slice<TileEdge> &edges) :
             camera{camera}, columns{columns}, edges{edges} {}

    RenderData render_data;
    RenderState render_state;
    Dimensions dimensions;

    f32 up_aim, prior_up_aim;
    f32 up_aim_over_focal_length;
    u16 prior_screen_height;

    bool useGPU = false;

    WallHitGroup wall_hits[MAX_WALL_HITS_COUNT];
    GroundHit ground_hits[MAX_GROUND_HITS_COUNT];

    void init() {
        render_state.init();

        portals.from.init();
        portals.to.init();

        texel_size = 4.0f / (f32)render_data.textures[0].width;
        last_mip = (u8)(render_data.textures[0].mip_count - 1);

        prior_screen_height = 0;
        prior_up_aim = 0.0f;

        initDataOnGPU(render_data);
    }

    void onScreenChanged() {
        forward = vec2(-camera.orientation.Z.x, -camera.orientation.Z.z).normalized();
        right_step = vec2(camera.orientation.X.x, camera.orientation.X.z).normalized() / (f32)screen_height;
        column_height_factor = 2.0f * camera.focal_length * (f32)screen_height;
        pixel_coverage_factor = 2.0f * camera.focal_length / (f32)screen_height;
        first_ray_direction = camera.focal_length * forward + right_step * (0.5f - 0.5f * (f32)screen_width);
        up_aim = camera.orientation.Z.y;
        up_aim_over_focal_length = up_aim / camera.focal_length;
        mid_point = (i32)((1.0f + up_aim) * (f32)(screen_height >> 1));

        generateWallHits();
        if (prior_screen_height != screen_height ||
            prior_up_aim != up_aim)
            generateFloorAndCeilingHits();

        prior_up_aim = up_aim;
    }

    void onResize() {
        screen_height = render_state.screen_height = (dimensions.height >> 1) << 1;
        screen_width = render_state.screen_width = dimensions.width;
        onScreenChanged();

        prior_screen_height = screen_height;
    }

    void generateFloorAndCeilingHits() {
        f32 Y = 1.0f + up_aim;

        f32 screen_pixel_height = 2.0f / (f32)screen_height;

        f32 Z, priorZ = 1.0f / (Y + screen_pixel_height);
        i32 y = 0;

        for (; y < mid_point; y++, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z * 2.0f;
            ground_hits[y].mip = computeMip(Z - priorZ, texel_size, last_mip);
            priorZ = Z;
        }

        Y = 1.0f - up_aim;
        priorZ = 1.0f / (Y + screen_pixel_height);
        y = screen_height - 1;

        for (; y > mid_point; y--, Y -= screen_pixel_height) {
            Z = 1.0f / Y;
            ground_hits[y].z = Z * 2.0f;
            ground_hits[y].mip = computeMip(Z - priorZ, texel_size, last_mip);
            priorZ = Z;
        }

        uploadGroundHits(ground_hits, screen_height);
    }

    void generateWallHits() {
        if (useGPU) {
            generateWallHitsOnGPU(*this);
            downloadWallHits(wall_hits, screen_width);
        } else {
            WallHitGroup wall_hit_group;
            vec2 ray_direction = first_ray_direction;
            for (u16 x = 0; x < screen_width; x++, ray_direction += right_step) {
                generateWallHit(wall_hit_group, ray_direction, edges, columns);
                wall_hits[x] = wall_hit_group;
            }
        }
    }

    void toggleUseOfGPU() {
#ifdef __CUDACC__
        if (useGPU) {
            downloadWallHits(wall_hits, screen_width);
            useGPU = false;
            if (render_state.step_count > 4)
                render_state.step_count = 4;

            render_state.flags = render_state.flags & ~VOLUMETRIC;
            render_state.flags = render_state.flags & ~VOLUMETRIC_SHADOWS;
        } else {
            uploadWallHits(wall_hits, screen_width);
            uploadColumns(columns);
            uploadEdges(edges);
            useGPU = true;
        }
#endif
    }

    void updatePortalLights() {
        for (u8 i = 0; i < render_state.light_count; i++) {
            vec2 light_position{render_state.lights[i].position.x, render_state.lights[i].position.z};

            i32 ray_rotation = portals.from.getRotation(portals.to.edge_is);
            vec2 origin_to_portal = vec2{portals.from.position.x, portals.from.position.z} - light_position;
            if (ray_rotation == 90) {
                origin_to_portal = origin_to_portal.ccw90();
            } else if (ray_rotation == -90) {
                origin_to_portal = origin_to_portal.cw90();
            } else if (ray_rotation == 180) {
                origin_to_portal = -origin_to_portal;
            }
            vec2 target = vec2{portals.to.position.x, portals.to.position.z} - origin_to_portal;
            render_state.lights_through_portal_from[i].x = target.x;
            render_state.lights_through_portal_from[i].z = target.y;
            render_state.lights_through_portal_from[i].y = render_state.lights[i].position.y;

            ray_rotation = portals.to.getRotation(portals.from.edge_is);
            origin_to_portal = vec2{portals.to.position.x, portals.to.position.z} - light_position;
            if (ray_rotation == 90) {
                origin_to_portal = origin_to_portal.ccw90();
            } else if (ray_rotation == -90) {
                origin_to_portal = origin_to_portal.cw90();
            } else if (ray_rotation == 180) {
                origin_to_portal = -origin_to_portal;
            }
            target = vec2{portals.from.position.x, portals.from.position.z} - origin_to_portal;
            render_state.lights_through_portal_to[i].x = target.x;
            render_state.lights_through_portal_to[i].z = target.y;
            render_state.lights_through_portal_to[i].y = render_state.lights[i].position.y;
        }
    }

    void renderOnCPU(u32* window_content) {
        PixelShader pixel_shader{render_data, render_state, edges, columns, portals};
        u32 offset = 0;
        for (u16 y = 0; y < screen_height; y++) {
            for (u16 x = 0; x < screen_width; x++, offset++) {
                window_content[offset] = pixel_shader.shade(
                    ground_hits,
                    wall_hits,
                    position,
                    x,
                    y,
                    mid_point).asContent();
            }
        }
    }

    void render(u32* window_content) {
#ifdef __CUDACC__
        if (useGPU) renderOnGPU(*this, render_state, window_content);
        else        renderOnCPU(window_content);
#else
        renderOnCPU(window_content);
#endif
    }
};