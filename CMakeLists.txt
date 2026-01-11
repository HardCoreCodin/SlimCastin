cmake_minimum_required(VERSION 3.11)

project(DungeonCrawler)
add_executable(DungeonCrawler WIN32 src/examples/DungeonCrawler.cpp)

project(bmp2texture)
add_executable(bmp2texture src/bmp2texture.cpp)

include(CheckLanguage)
check_language(CUDA)
if(DEFINED CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    set(CUDA_VERBOSE_BUILD ON)

    project(DungeonCrawler_XPU CUDA)
    add_executable(DungeonCrawler_XPU WIN32 src/examples/DungeonCrawler.cu)
    set_target_properties(DungeonCrawler_XPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

endif()


#add_compile_options(
#        $<$<CONFIG:RELEASE>:-O3>
#        $<$<CONFIG:RELEASE>:-Oi>
#        $<$<CONFIG:RELEASE>:-Mt>
#        $<$<CONFIG:DEBUG>:-JMC>
#        $<$<CONFIG:DEBUG>:-MTd>
#        $<$<CONFIG:DEBUG>:-Z7>
#        -WX-
#        -W4
#        -wd4100
#        -GR-
#        -EHa-
#        -GL
#        -GS
#        -FC
#        -permissive-
#        -sdl
#        -nologo
#        -analyze-
#        -fp:fast
#        -Zc:inline
#        -diagnostics:caret
#        -errorReport:none
#)


#target_compile_options(07_Meshes PUBLIC
#        $<$<CONFIG:RELEASE>:-O3>
#        $<$<CONFIG:RELEASE>:-Oi>
#        $<$<CONFIG:RELEASE>:-Mt>
#        $<$<CONFIG:DEBUG>:-JMC>
#        $<$<CONFIG:DEBUG>:-MTd>
#        $<$<CONFIG:DEBUG>:-Z7>
#        -WX-
#        -W4
#        -wd4100
#        -GR-
#        -EHa-
#        -GL
#        -GS
#        -FC
#        -permissive-
#        -sdl
#        -nologo
#        -analyze-
#        -fp:fast
#        -Zc:inline
#        -diagnostics:caret
#        -errorReport:none
#        )