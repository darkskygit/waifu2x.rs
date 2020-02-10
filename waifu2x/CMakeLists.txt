project(waifu2x-ncnn-vulkan)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)

find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

macro(compile_shader SHADER_SRC)
    set(SHADER_SRC_FULLPATH ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC})

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -V -s -e ${SHADER_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -V -s -e ${SHADER_fp16s_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

    if(NOT NO_INT8_SUPPORT)
    # int8 storage
    set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")

    set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -e ${SHADER_int8s_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_int8s_SPV_HEX_FILE})
    endif(NOT NO_INT8_SUPPORT)
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# change these path to yours
include_directories(${Vulkan_INCLUDE_DIR})
include_directories(${INCLUDE_LIST})
link_directories(${LINK_LIST})

# look for vulkan compute shader and compile
set(SHADER_SPV_HEX_FILES)

compile_shader(waifu2x_preproc.comp)
compile_shader(waifu2x_postproc.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

if(NO_INT8_SUPPORT)
    add_compile_definitions(NO_INT8_SUPPORT)
endif(NO_INT8_SUPPORT)

if(WAIFU2X_NOISE_ONLY)
    add_compile_definitions(WAIFU2X_NOISE_ONLY)
endif(WAIFU2X_NOISE_ONLY)

if(WAIFU2X_UPCONV7_ONLY)
    add_compile_definitions(WAIFU2X_UPCONV7_ONLY)
endif(WAIFU2X_UPCONV7_ONLY)

if (WAIFU2X_FULL)
endif(WAIFU2X_FULL)

if (MSVC AND MSVC_STATIC)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endif (MSVC AND MSVC_STATIC)

add_library(waifu2x-ncnn-vulkan STATIC waifu2x.cpp)

add_dependencies(waifu2x-ncnn-vulkan generate-spirv)

if(MSVC AND CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
    set_target_properties(waifu2x-ncnn-vulkan PROPERTIES COMPILE_FLAGS "/GL")
    set_target_properties(waifu2x-ncnn-vulkan PROPERTIES LINK_FLAGS "/LTCG")
endif()

target_link_libraries(waifu2x-ncnn-vulkan ncnn ${Vulkan_LIBRARYS} ${OpenMP_CXX_LIBRARIES})

install(TARGETS waifu2x-ncnn-vulkan ARCHIVE DESTINATION lib)