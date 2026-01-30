#pragma once

#include "tilemap_base.h"

#include <unordered_map>



// struct TopLeft2Df { f32 left, top; };
// struct TopLeft2Di { i32 left, top; };
// struct BottomRight2Df { f32 right, bottom; };
// struct BottomRight2Di { i32 right, bottom; };

// union TopLeft2DfMin { vec2 min; TopLeft2Df top_left; };
// union TopLeft2DiMin { vec2i min; TopLeft2Di top_left; };
// union BottomRight2DfMax { vec2 max;  BottomRight2Df bottom_right; };
// union BottomRight2DiMax { vec2i max; BottomRight2Di bottom_right; };

// struct Bounds2Df { TopLeft2Df tl; BottomRight2Df br; };
// struct Bounds2Di { TopLeft2Di tl; BottomRight2Di br; };



// bool inBounds(const Bounds2Df &b, const vec2 &p) { return inRange(b.tl.left, p.x, b.br.right) && inRange(b.tl.top, p.y, b.br.bottom); }
// bool inBounds(const Bounds2Di &b, const vec2i &p) { return inRange(b.tl.left, p.x, b.br.right) && inRange(b.tl.top, p.y, b.br.bottom); }

// Rect2Df :: struct {using bounds: Bounds2Df, using size: Size2Df, position: vec2}
// Rect2Di :: struct {using bounds: Bounds2Di, using size: Size2Di, position: vec2i}




struct TileEdge {
	vec2i from{};
	vec2i to{};
	i32 length = 0;
	// i32 portal_ray_rotation = 0;
	// TileEdge* portal_to;
	// bool portal_edge_dir_flip;
	u8 texture_id = 0;
	u8 is = 0;
};


struct TileSide {
	// TileSide* portal_to;
	// TileSide* portal_from;
	u8 texture_id = 0;
	u16 edge_id = (u16)(-1);
};


struct Tile {
	TileSide top, bottom, left, right;

	// Bounds2Di bounds;

	bool
	is_full,
	has_left_edge,
	has_right_edge,
	has_top_edge,
	has_bottom_edge;
};

typedef Slice<Tile> TileRow;


struct TileMap : Grid<Tile> {
	Slice<Circle> columns;
	Slice<TileEdge> edges;
	Slice<LocalEdge> local_edges;

	// i32 vertex_count;
	// Slice<vec2i> vertices;
	// Slice<vec2> vertices_in_local_space;

	u8 columns_texture_id;

	i32 portal_sides_count;
	// Slice<TileSide*> portal_sides;
	std::unordered_map<TileSide*, Tile*> side_to_tile;

	TileSide* all_portal_sides[MAX_TILE_MAP_EDGES];
	TileRow all_rows[MAX_TILE_MAP_HEIGHT];
	Tile all_tiles[MAX_TILE_MAP_SIZE];
	TileEdge all_edges[MAX_TILE_MAP_EDGES];
	LocalEdge all_local_edges[MAX_TILE_MAP_EDGES];
	Circle all_columns[MAX_COLUMN_COUNT];
};


void initTileSide(TileSide* ts) {
	// ts->portal_from = nullptr;
	// ts->portal_to = nullptr;
	ts->edge_id = (u16)(-1);
	ts->texture_id = 0;
}


void initTile(Tile* t) {
	initTileSide(&t->top);
	initTileSide(&t->bottom);
	initTileSide(&t->left);
	initTileSide(&t->right);

	t->is_full = false;

	t->has_left_edge = false;
	t->has_right_edge = false;
	t->has_top_edge = false;
	t->has_bottom_edge = false;

	// t->bounds.tl.top = 0;
	// t->bounds.tl.left = 0;
	// t->bounds.br.bottom = 0;
	// t->bounds.br.right = 0;
}


void initTileMap(TileMap& tm, u16 Width = MAX_TILE_MAP_WIDTH, u16 Height = MAX_TILE_MAP_HEIGHT) {
	tm.width = Width;
	tm.height = Height;
	tm.columns_texture_id = 0;
	Slice<Tile> all_tiles;
	setSliceToStaticArray(all_tiles, tm.all_tiles);
	for (int i = 0; i < MAX_TILE_MAP_SIZE; i++) initTile(all_tiles.data + i);
	setSliceToStaticArray(tm.columns, tm.all_columns);
	setSliceToStaticArray(tm.edges, tm.all_edges);
	setSliceToStaticArray(tm.local_edges, tm.all_local_edges);
	// setSliceToStaticArray(tm.portal_sides, tm.all_portal_sides);
	initGrid<Tile>(tm, Width, Height, all_tiles);
}


void readTileMap(TileMap& tm, Slice<Tile*> map_grid) {
	u32 offset = 0;
    // Bounds2Di current_bounds;

	// current_bounds.tl.top = 0;
	// current_bounds.tl.left = 0;
	// current_bounds.br.bottom = 1;
	// current_bounds.br.right = 1;

	std::unordered_map<TileSide*, TileSide*> cell_side_to_tile_side;
	// bool has_portals = false;
	tm.portal_sides_count = 0;

	Slice<Tile>* row = nullptr;
	Tile* tile = nullptr;

	iterSlice(tm.cells, row, y) {
		iterSlice((*row), tile, x) {
			tm.side_to_tile[&tile->left] = tile;
			tm.side_to_tile[&tile->right] = tile;
			tm.side_to_tile[&tile->top] = tile;
			tm.side_to_tile[&tile->bottom] = tile;
		}
	}

	iterSlice(tm.cells, row, y) {
		iterSlice((*row), tile, x) {
			initTile(tile);
			// tile->bounds = current_bounds;

			Tile* map_cell = map_grid[offset];
			tile->is_full = map_cell != nullptr;
			if (tile->is_full) {
				tile->left = map_cell->left;
				tile->right = map_cell->right;
				tile->top = map_cell->top;
				tile->bottom = map_cell->bottom;

				cell_side_to_tile_side[&map_cell->left] = &tile->left;
				cell_side_to_tile_side[&map_cell->right] = &tile->right;
				cell_side_to_tile_side[&map_cell->top] = &tile->top;
				cell_side_to_tile_side[&map_cell->bottom] = &tile->bottom;

				// if (map_cell->left.portal_to != nullptr) {
				// 	tm.portal_sides[tm.portal_sides_count] = &tile->left;
				// 	tm.portal_sides_count += 1;
				// }
				// if (map_cell->right.portal_to != nullptr) {
				// 	tm.portal_sides[tm.portal_sides_count]= &tile->right;
				// 	tm.portal_sides_count += 1;
				// }
				// if (map_cell->top.portal_to != nullptr) {
				// 	tm.portal_sides[tm.portal_sides_count]= &tile->top;
				// 	tm.portal_sides_count += 1;
				// }
				// if (map_cell->bottom.portal_to != nullptr) {
				// 	tm.portal_sides[tm.portal_sides_count]= &tile->bottom;
				// 	tm.portal_sides_count += 1;
				// }
			} else {
				tile->is_full = false;
			}

			// current_bounds.tl.left += 1;
			// current_bounds.br.right += 1;
			offset += 1;
		}

  //       current_bounds.tl.left = 0;
		// current_bounds.br.right = 1;
		// current_bounds.tl.top += 1;
		// current_bounds.br.bottom += 1;
    }

	// if (tm.portal_sides_count != 0) {
	// 	offset = 0;
	// 	iterSlice(tm.cells, row, y) {
	// 		iterSlice((*row), tile, x) {
	// 			Tile* map_cell = map_grid[offset];
	// 			if (map_cell) {
	// 				if (map_cell->left.portal_to != nullptr) {
	// 					tile->left.portal_to = cell_side_to_tile_side[map_cell->left.portal_to];
	// 					tile->left.portal_to->portal_from = &tile->left;
	// 				}
	// 				if (map_cell->right.portal_to != nullptr) {
	// 					tile->right.portal_to = cell_side_to_tile_side[map_cell->right.portal_to];
	// 					tile->right.portal_to->portal_from = &tile->right;
	// 				}
	// 				if (map_cell->top.portal_to != nullptr) {
	// 					tile->top.portal_to = cell_side_to_tile_side[map_cell->top.portal_to];
	// 					tile->top.portal_to->portal_from = &tile->top;
	// 				}
	// 				if (map_cell->bottom.portal_to != nullptr) {
	// 					tile->bottom.portal_to = cell_side_to_tile_side[map_cell->bottom.portal_to];
	// 					tile->bottom.portal_to->portal_from = &tile->bottom;
	// 				}
	// 			}
	//
	// 			offset += 1;
	// 		}
	// 	}
	// }
}


void moveTileMap(TileMap& tm, const vec2& origin) {
	LocalEdge local_edge;
	TileEdge* edge = nullptr;
	tm.local_edges.size = 0;
	iterSlice(tm.edges, edge, i) {
		local_edge.texture_id = edge->texture_id;
		local_edge.from = vec2((f32)edge->from.x - origin.x, (f32)edge->from.y - origin.y);
		local_edge.to = vec2((f32)edge->to.x - origin.x, (f32)edge->to.y - origin.y);
		local_edge.is = edge->is;

		if (local_edge.is & (FACING_LEFT | FACING_RIGHT)) {
			if (local_edge.from.x > 0) local_edge.is |= ON_THE_RIGHT;
			if (local_edge.to.x   < 0) local_edge.is |= ON_THE_LEFT;
		} else {
			if (local_edge.to.y   < 0) local_edge.is |= ABOVE;
			if (local_edge.from.y > 0) local_edge.is |= BELOW;
		}
		if (local_edge.is & FACING_LEFT  && local_edge.is & ON_THE_RIGHT ||
			local_edge.is & FACING_RIGHT && local_edge.is & ON_THE_LEFT ||
			local_edge.is & FACING_DOWN  && local_edge.is & ABOVE ||
			local_edge.is & FACING_UP    && local_edge.is & BELOW)
			tm.local_edges.data[tm.local_edges.size++] = local_edge;
	}
}


struct TileCheck {
	bool exists;
	Tile* tile;
	Slice<Tile> row;
};


void generateTileMapEdges(TileMap& tm) {
	TileCheck above, below, left, right;

	vec2i position;
	tm.edges.size = 0;

	Slice<Tile>* row = nullptr;
	Tile* current_tile = nullptr;

	Slice<Tile> _{nullptr, 0};

	iterSlice(tm.cells, row, y) {
		above.exists = y > 0;
		below.exists = (i32)y < tm.height - 1;

		above.row = above.exists ? tm.cells[y - 1] : _;
		below.row = below.exists ? tm.cells[y + 1] : _;

		iterSlice((*row), current_tile, x) {
        	left.exists  = x > 0;
        	right.exists = (i32)x < (tm.width - 1);

        	left.tile  = left.exists  ? &(*row)[x - 1] : nullptr;
        	right.tile = right.exists ? &(*row)[x + 1] : nullptr;
        	above.tile = above.exists ? &above.row[x] : nullptr;
        	below.tile = below.exists ? &below.row[x] : nullptr;

        	if (current_tile->is_full) {
				current_tile->has_left_edge   = left.exists  &&  !left.tile->is_full;
	        	current_tile->has_right_edge  = right.exists && !right.tile->is_full;
	        	current_tile->has_top_edge    = above.exists && !above.tile->is_full;
	        	current_tile->has_bottom_edge = below.exists && !below.tile->is_full;

	        	if (current_tile->has_left_edge) { // Create/extend left edge:
		        	if (above.exists && above.tile->has_left_edge) {// &&
					    // above.tile->left.portal_to == current_tile->left.portal_to && current_tile->left.portal_from == nullptr) { // Tile above has a left edge, extend it:
		        		current_tile->left.edge_id = above.tile->left.edge_id;
		        		TileEdge& left_edge = tm.edges[current_tile->left.edge_id];
		        		left_edge.length++;
		        		left_edge.to.y++;
		        	} else { // No left edge above - create new one:
		        		current_tile->left.edge_id = (u16)tm.edges.size;
		        		TileEdge& left_edge = tm.edges.data[tm.edges.size++];
		        		left_edge.is = FACING_LEFT;
						left_edge.texture_id = current_tile->left.texture_id;
		        		left_edge.to = left_edge.from = position;
		        		left_edge.to.y++;
			        }
			    }

				if (current_tile->has_right_edge) { // Create/extend right edge:
		        	if (above.exists && above.tile->has_right_edge) {// &&
					    // above.tile->right.portal_to == current_tile->right.portal_to && current_tile->right.portal_from == nullptr) { // Tile above has a right edge, extend it:
		        		current_tile->right.edge_id = above.tile->right.edge_id;
		        		TileEdge& right_edge = tm.edges.data[above.tile->right.edge_id];
		        		right_edge.length++;
		        		right_edge.to.y++;
		        	} else { // No right edge above - create new one:
		        		current_tile->right.edge_id = (u16)tm.edges.size;
		        		TileEdge& right_edge = tm.edges[tm.edges.size++];
		        		right_edge.is = FACING_RIGHT;
						right_edge.texture_id = current_tile->right.texture_id;
		        		right_edge.from = right_edge.to = position;
		        		right_edge.from.x++;
		        		right_edge.to.x++;
		        		right_edge.to.y++;
			        }
				}

		        if (current_tile->has_top_edge) { // Create/extend top edge:
		        	if (left.exists && left.tile->has_top_edge) {// &&
						// left.tile->top.portal_to == current_tile->top.portal_to && current_tile->top.portal_from == nullptr) { // Tile on the left has a top edge, extend it:
		        		current_tile->top.edge_id = left.tile->top.edge_id;
		        		TileEdge& top_edge = tm.edges[left.tile->top.edge_id];
		        		top_edge.length++;
		        		top_edge.to.x++;
		        	} else { // No top edge on the left - create new one:
		        		current_tile->top.edge_id = (u16)tm.edges.size;
		        		TileEdge& top_edge = tm.edges.data[tm.edges.size++];
		        		top_edge.is = FACING_UP;
		        		top_edge.texture_id = current_tile->top.texture_id;
		        		top_edge.from = top_edge.to = position;
		        		top_edge.to.x++;
			        }
		        }

		        if (current_tile->has_bottom_edge) { // Create/extend bottom edge:
		        	if (left.exists && left.tile->has_bottom_edge) { // &&
						// left.tile->bottom.portal_to == current_tile->bottom.portal_to && current_tile->bottom.portal_from == nullptr) {// Tile on the left has a bottom edge, extend it:
		        		current_tile->bottom.edge_id = left.tile->bottom.edge_id;
		        		TileEdge& bottom_edge = tm.edges[left.tile->bottom.edge_id];
		        		bottom_edge.length++;
		        		bottom_edge.to.x++;
		        	} else { // No bottom edge on the left - create new one:
		        		current_tile->bottom.edge_id = (u16)tm.edges.size;
		        		TileEdge& bottom_edge = tm.edges.data[tm.edges.size++];
		        		bottom_edge.is = FACING_DOWN;
		        		bottom_edge.texture_id = current_tile->bottom.texture_id;
		        		bottom_edge.from = bottom_edge.to = position;
		        		bottom_edge.from.y++;
		        		bottom_edge.to.x++;
		        		bottom_edge.to.y++;
			        }
	        	}
        	} else {
        		current_tile->has_left_edge   = false;
	        	current_tile->has_right_edge  = false;
	        	current_tile->has_top_edge    = false;
	        	current_tile->has_bottom_edge = false;
        	}

	        // current_tile->bounds.tl.left = position.x;
	        // current_tile->bounds.br.right = position.x + 1;

	        // current_tile->bounds.tl.top = position.y;
	        // current_tile->bounds.br.bottom = position.y + 1;

			position.x += 1;
        }

        position.x  = 0;
        position.y += 1;
    }

	// setSliceToRangeOfStaticArray(tm.edges, tm.all_edges, 0, tm.edges.size);
	// if (tm.portal_sides_count) {
	// 	for (i32 i = 0; i < tm.portal_sides_count; i++) {
	// 		const TileSide& side = *tm.portal_sides[i];
	// 		TileEdge& from_edge = *side.edge;
	// 		TileEdge& to_edge = *side.portal_to->edge;
	// 		from_edge.portal_to = &to_edge;
	//
	// 		//
	//
	// 		if (from_edge.is_vertical) {
	// 			if (to_edge.is_horizontal) {
	// 				from_edge.portal_edge_dir_flip = to_edge.is_facing_up;
	// 				if (from_edge.is_facing_right) {
	// 					from_edge.portal_ray_rotation = to_edge.is_facing_up ? 90 : -90; // to_edge.is_facing_down
	// 				} else {
	// 					from_edge.portal_ray_rotation = to_edge.is_facing_down ? 90: -90; // to_edge.is_facing_up
	// 				}
	// 			} else {
	// 				from_edge.portal_edge_dir_flip = from_edge.is_facing_right != to_edge.is_facing_right;
	// 				from_edge.portal_ray_rotation = from_edge.portal_edge_dir_flip ? 180 : 0;
	// 			}
	// 		} else {
	// 			if (to_edge.is_vertical) {
	// 				from_edge.portal_edge_dir_flip = to_edge.is_facing_left;
	// 				if (from_edge.is_facing_up) {
	// 					from_edge.portal_ray_rotation = to_edge.is_facing_left ? 90 : -90; // to_edge.is_facing_right
	// 				} else {
	// 					from_edge.portal_ray_rotation = to_edge.is_facing_right ? 90 : -90; // to_edge.is_facing_left
	// 				}
	// 			} else {
	// 				from_edge.portal_edge_dir_flip = from_edge.is_facing_up != to_edge.is_facing_up;
	// 				from_edge.portal_ray_rotation = from_edge.portal_edge_dir_flip ? 180 : 0;
	// 			}
	// 		}
	// 	}
	// }
}