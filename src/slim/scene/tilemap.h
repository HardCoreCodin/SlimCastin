#pragma once

#include "tilemap_base.h"

#include <unordered_map>

#include "../renderer/render_data.h"


struct TileSide {
	u8 texture_id = 0;
	u16 edge_id = (u16)(-1);
};


struct Tile {
	TileSide top, bottom, left, right;

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

	u8 columns_texture_id;
	std::unordered_map<TileSide*, Tile*> side_to_tile;

	TileSide* all_portal_sides[MAX_TILE_MAP_EDGES];
	TileRow all_rows[MAX_TILE_MAP_HEIGHT];
	Tile all_tiles[MAX_TILE_MAP_SIZE];
	TileEdge all_edges[MAX_TILE_MAP_EDGES];
	Circle all_columns[MAX_COLUMN_COUNT];
};


void initTileSide(TileSide* ts) {
	ts->edge_id = INVALID_EDGE_ID;
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
}


void initTileMap(TileMap& tm, u16 Width = MAX_TILE_MAP_WIDTH, u16 Height = MAX_TILE_MAP_HEIGHT) {
	tm.width = Width;
	tm.height = Height;
	tm.columns_texture_id = 0;

	for (int i = 0; i < MAX_TILE_MAP_SIZE; i++) initTile(tm.all_tiles + i);
	setSliceToStaticArray(tm.columns, tm.all_columns);
	setSliceToStaticArray(tm.edges, tm.all_edges);

	tm.columns.size = tm.edges.size = 0;
	initGrid<Tile>(tm, Width, Height, {&tm.all_tiles[0], ARRAY_SIZE(tm.all_tiles)});
}


void readTileMap(TileMap& tm, Slice<Tile*> map_grid) {
	u32 offset = 0;
	std::unordered_map<TileSide*, TileSide*> cell_side_to_tile_side;

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
			} else {
				tile->is_full = false;
			}
			offset += 1;
		}
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
		        		current_tile->left.edge_id = above.tile->left.edge_id;
		        		TileEdge& left_edge = tm.edges[current_tile->left.edge_id];
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
		        		current_tile->right.edge_id = above.tile->right.edge_id;
		        		TileEdge& right_edge = tm.edges.data[above.tile->right.edge_id];
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
		        		current_tile->top.edge_id = left.tile->top.edge_id;
		        		TileEdge& top_edge = tm.edges[left.tile->top.edge_id];
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
		        		current_tile->bottom.edge_id = left.tile->bottom.edge_id;
		        		TileEdge& bottom_edge = tm.edges[left.tile->bottom.edge_id];
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
			position.x += 1;
        }

        position.x  = 0;
        position.y += 1;
    }
}