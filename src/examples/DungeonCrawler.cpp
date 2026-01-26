#include "./textures.h"

#include "../slim/renderer/renderer.h"
#include "../slim/draw/hud.h"
#include "../slim/app.h"



// WALLS := `
// 11111111111111111111111111111111
// 1_________1____________________1
// 1__1___11_1____________________1
// 111111_1111____________________1
// 1__1_____11____________________1
// 1____11___1____________________1
// 1_____1_111____________________1
// 1_11111__11____________________1
// 1_________1____________________1
// 11111111111____________________1
// 1______________________________1
// 1___________11111111___________1
// 1__________________1___________1
// 1__________________1___________1
// 1__________________1___________1
// 1__________________11111111____1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 1______________________________1
// 11111111111111111111111111111111
// `;

TileSide TILE_SIDE{};
Tile W_TILE{TILE_SIDE, TILE_SIDE, TILE_SIDE, TILE_SIDE, Bounds2Di{}, true, true, true, true, true};
Tile F_TILE{TILE_SIDE, TILE_SIDE, TILE_SIDE, TILE_SIDE, Bounds2Di{}, true, true, true, true, true};
Tile T_TILE{TILE_SIDE, TILE_SIDE, TILE_SIDE, TILE_SIDE, Bounds2Di{}, true, true, true, true, true};

Tile* F{&F_TILE};
Tile* T{&T_TILE};
Tile* W{&W_TILE};
Tile* I{nullptr};

Tile* WALLS[] = {
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
	W,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,W,I,I,I,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,I,W,W,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,W,I,I,I,I,I,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,W,W,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,W,I,W,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,W,W,W,W,W,I,I,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,W,W,W,W,W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,W,W,W,W,W,W,W,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,W,W,W,W,W,W,W,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
};
Tile* WALLS2[] = {
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,I,W,
	W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,
};


struct DungeonCrawler : SlimApp {
    bool use_gpu = USE_GPU_BY_DEFAULT;
    bool antialias = false;

    // HUD:
    HUDLine FPS {"FPS : "};
    HUDLine GPU {"GPU : ", "Off", "On", &use_gpu};
    HUDLine AA  {"AA  : ", "Off", "On", &antialias};
    HUDLine Mode{"Mode: ", "Beauty"};
    HUD hud{{4}, &FPS};


	// Viewport:
	Camera camera{{0, 0 * DEG_TO_RAD, 0}, {13, 0, 3}};
	Canvas canvas;
	Viewport viewport{canvas, &camera};

	TileMap tile_map;
	Slice<Texture> textures_slice{textures, Texture_Count};

	RayCasterSettings settings;

	bool initted = false;

    void OnUpdate(f32 delta_time) override {
        i32 fps = (i32)render_timer.average_frames_per_second;
        FPS.value = fps;
        FPS.value_color = fps >= 60 ? Green : (fps >= 24 ? Cyan : (fps < 12 ? Red : Yellow));

		bool tile_map_changed = false;

        if (!controls::is_pressed::alt) {
	        viewport.updateNavigation(delta_time);
        	if (viewport.navigation.moved || tile_map_changed) ray_cast_renderer::onMove(*viewport.camera, tile_map);
        	if (viewport.navigation.moved || tile_map_changed ||
	            viewport.navigation.turned) ray_cast_renderer::onMoveOrTurn(camera, tile_map);
        }
    }

    void OnRender() override {
        ray_cast_renderer::render(canvas, use_gpu);

        if (hud.enabled)
            drawHUD(hud, canvas);

        canvas.drawToWindow();
    }

    void OnKeyChanged(u8 key, bool is_pressed) override {
        if (!is_pressed) {
            if (key == controls::key_map::tab) hud.enabled = !hud.enabled;
            if (key == 'G' && USE_GPU_BY_DEFAULT) use_gpu = !use_gpu;
            if (key == 'V') {
                antialias = !antialias;
                canvas.antialias = antialias ? SSAA : NoAA;
            }
            if (key == '1') settings.render_mode = RenderMode_Beauty;
            if (key == '2') settings.render_mode = RenderMode_Depth;
            if (key == '3') settings.render_mode = RenderMode_MipLevel;
            if (key == '4') settings.render_mode = RenderMode_UVs;
            const char* mode;
            switch (settings.render_mode) {
                case RenderMode_Beauty:    mode = "Beauty"; break;
                case RenderMode_Depth:     mode = "Depth"; break;
                case RenderMode_MipLevel:  mode = "Mip Level"; break;
                case RenderMode_UVs:       mode = "UVs"; break;
            }
            Mode.value.string = mode;
        }
        Move &move = viewport.navigation.move;
        Turn &turn = viewport.navigation.turn;
        if (key == 'Q') turn.left     = is_pressed;
        if (key == 'E') turn.right    = is_pressed;
        if (key == 'R') move.up       = is_pressed;
        if (key == 'F') move.down     = is_pressed;
        if (key == 'W') move.forward  = is_pressed;
        if (key == 'S') move.backward = is_pressed;
        if (key == 'A') move.left     = is_pressed;
        if (key == 'D') move.right    = is_pressed;
    }

    void OnWindowResize(u16 width, u16 height) override {
        viewport.updateDimensions(width, height);
        canvas.dimensions.update(width, height);

    	if (initted) ray_cast_renderer::onResize(width, height, *viewport.camera, tile_map);
    	else {
    		initted = true;

    		// F->bottom.portal_to = &T->right;
    		// T->right.portal_to = &F->bottom;

    		initTileMap(tile_map);
    		readTileMap(tile_map, SliceFromStaticArray(Tile*, WALLS));
    		generateTileMapEdges(tile_map);

    		settings.init(textures_slice, Texture_ColoredStone, Texture_RedStone, tile_map.width, tile_map.height);

    		ray_cast_renderer::init(&settings, viewport.dimensions, *viewport.camera, tile_map);
    	}
    }

    void OnMouseButtonDown(mouse::Button &mouse_button) override {
        mouse::pos_raw_diff_x = mouse::pos_raw_diff_y = 0;
    }

    void OnMouseButtonDoubleClicked(mouse::Button &mouse_button) override {
        if (&mouse_button == &mouse::left_button) {
            mouse::is_captured = !mouse::is_captured;
            os::setCursorVisibility(!mouse::is_captured);
            os::setWindowCapture(    mouse::is_captured);
            OnMouseButtonDown(mouse_button);
        }
    }
};

SlimApp* createApp() {
	return (SlimApp*)new DungeonCrawler();
}