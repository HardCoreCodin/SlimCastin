#pragma once

#include "../slim/serialization/texture.h"
#include "../slim/core/string.h"


const char* floor_texture_files[] = {
    "colored_stone.texture",
    "purple_stone.texture"
};
const char* wall_texture_files[] = {
    "cobblestone2.texture",
    "red_stone.texture"
};

enum FloorTextureID {
    Floor_ColoredStone,
    Floor_PurpleStone,

    Floor_TextureCount
};
enum WallTextureID {
    Wall_CobbleStone,
    Wall_RedStone,

    Wall_TextureCount
};

Texture floor_textures[Floor_TextureCount];
Texture wall_textures[Wall_TextureCount];

char floor_textures_string_buffers[Floor_TextureCount][200]{};
char wall_textures_string_buffers[Wall_TextureCount][200]{};

String floor_texture_strings[Floor_TextureCount] {
    {floor_textures_string_buffers[0], 200},
    {floor_textures_string_buffers[1], 200}
};
String wall_texture_strings[Wall_TextureCount] {
    {wall_textures_string_buffers[0], 200},
    {wall_textures_string_buffers[1], 200}
};

TexturePack floor_texture_pack{Floor_TextureCount, floor_textures, floor_texture_strings, floor_texture_files, __FILE__, Terabytes(3)};
TexturePack wall_texture_pack{Wall_TextureCount, wall_textures, wall_texture_strings, wall_texture_files, __FILE__, Terabytes(3) + Megabytes(3)};