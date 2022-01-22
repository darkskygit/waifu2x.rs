use cmake::Config;
use path_absolutize::Absolutize;
use std::env;
use std::fs::create_dir;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let proj_dir = root_dir.absolutize().unwrap();
    let vulkan_dir = env::var("NCNN_VULKAN_DIR")
        .and_then(|path| Ok(PathBuf::from(path)))
        .unwrap_or(proj_dir.join("vulkan"));
    println!("vulkan_dir: {}", vulkan_dir.display());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ncnn_dir = out_dir.join("ncnn");
    let ncnn = if cfg!(feature = "bundled") && cfg!(windows) {
        use fs_extra::dir::{copy, CopyOptions};
        copy(
            proj_dir.join("ncnn").join("ncnn"),
            &out_dir,
            &CopyOptions::new(),
        )
        .unwrap_or_default();
        ncnn_dir
    } else {
        // md build && cd build
        // cmake -DNCNN_VULKAN=ON -DNCNN_BUILD_WITH_STATIC_CRT=ON -DNCNN_ENABLE_LTO=ON -DNCNN_STDIO=OFF -DNCNN_STRING=OFF ..
        // cmake --build . --config MinSizeRel -j 16
        // cmake --install . --config MinSizeRel
        let ncnn_proj_dir = out_dir.join("ncnn_proj");
        if !ncnn_proj_dir.exists() {
            if !Command::new("git")
                .args([
                    "clone",
                    "https://github.com/darkskygit/waifu2x.rs",
                    "--depth",
                    "1",
                    "--recursive",
                    "-j8",
                    "--single-branch",
                    "-b",
                    "ncnn",
                    "ncnn_proj",
                ])
                .current_dir(out_dir)
                .status()
                .unwrap()
                .success()
            {
                panic!("Failed to checkout ncnn");
            }
        }
        create_dir(&ncnn_dir).unwrap_or_default();
        let mut config = Config::new(ncnn_proj_dir);
        config
            .generator(if cfg!(windows) {
                "NMake Makefiles"
            } else {
                "Unix Makefiles"
            })
            .out_dir(ncnn_dir)
            .static_crt(true)
            .profile("MinSizeRel")
            .env("VULKAN_SDK", &vulkan_dir)
            .define("NCNN_BUILD_WITH_STATIC_CRT", "ON")
            .define("NCNN_ENABLE_LTO", "ON")
            .define("NCNN_VULKAN", "ON")
            .define("NCNN_STDIO", "OFF")
            .define("NCNN_STRING", "OFF")
            .define("WITH_LAYER_AbsVal", "OFF")
            .define("WITH_LAYER_ArgMax", "OFF")
            .define("WITH_LAYER_BatchNorm", "OFF")
            .define("WITH_LAYER_Bias", "OFF")
            .define("WITH_LAYER_BNLL", "OFF")
            .define("WITH_LAYER_Concat", "OFF")
            .define("WITH_LAYER_Dropout", "OFF")
            .define("WITH_LAYER_ELU", "OFF")
            .define("WITH_LAYER_Embed", "OFF")
            .define("WITH_LAYER_Exp", "OFF")
            .define("WITH_LAYER_Flatten", "OFF")
            .define("WITH_LAYER_Input", "OFF")
            .define("WITH_LAYER_Log", "OFF")
            .define("WITH_LAYER_LRN", "OFF")
            .define("WITH_LAYER_MemoryData", "OFF")
            .define("WITH_LAYER_MVN", "OFF")
            .define("WITH_LAYER_Power", "OFF")
            .define("WITH_LAYER_PReLU", "OFF")
            .define("WITH_LAYER_Proposal", "OFF")
            .define("WITH_LAYER_Reduction", "OFF")
            .define("WITH_LAYER_Reshape", "OFF")
            .define("WITH_LAYER_ROIPooling", "OFF")
            .define("WITH_LAYER_Sigmoid", "OFF")
            .define("WITH_LAYER_Slice", "OFF")
            .define("WITH_LAYER_Softmax", "OFF")
            .define("WITH_LAYER_SPP", "OFF")
            .define("WITH_LAYER_TanH", "OFF")
            .define("WITH_LAYER_Threshold", "OFF")
            .define("WITH_LAYER_Tile", "OFF")
            .define("WITH_LAYER_RNN", "OFF")
            .define("WITH_LAYER_LSTM", "OFF")
            .define("WITH_LAYER_BinaryOp", "OFF")
            .define("WITH_LAYER_UnaryOp", "OFF")
            .define("WITH_LAYER_ConvolutionDepthWise", "OFF")
            .define("WITH_LAYER_Padding", "OFF")
            .define("WITH_LAYER_Squeeze", "OFF")
            .define("WITH_LAYER_ExpandDims", "OFF")
            .define("WITH_LAYER_Normalize", "OFF")
            .define("WITH_LAYER_Permute", "OFF")
            .define("WITH_LAYER_PriorBox", "OFF")
            .define("WITH_LAYER_DetectionOutput", "OFF")
            .define("WITH_LAYER_Interp", "OFF")
            .define("WITH_LAYER_DeconvolutionDepthWise", "OFF")
            .define("WITH_LAYER_ShuffleChannel", "OFF")
            .define("WITH_LAYER_InstanceNorm", "OFF")
            .define("WITH_LAYER_Clip", "OFF")
            .define("WITH_LAYER_Reorg", "OFF")
            .define("WITH_LAYER_YoloDetectionOutput", "OFF")
            .define("WITH_LAYER_Quantize", "OFF")
            .define("WITH_LAYER_Dequantize", "OFF")
            .define("WITH_LAYER_Yolov3DetectionOutput", "OFF")
            .define("WITH_LAYER_PSROI'", "OFF")
            .define("WITH_LAYER_ROIAlign", "OFF")
            .define("WITH_LAYER_Packing", "OFF")
            .define("WITH_LAYER_Requantize", "OFF")
            .define("WITH_LAYER_Cast", "OFF")
            .define("WITH_LAYER_HardSigmoid", "OFF")
            .define("WITH_LAYER_SELU", "OFF")
            .define("WITH_LAYER_HardSwish", "OFF");
        if cfg!(target_os = "macos") && env::var("NCNN_VULKAN_DIR").is_ok() {
            let molten_vk = vulkan_dir.join("..").join("MoltenVK");
            config.define("Vulkan_INCLUDE_DIR", molten_vk.join("include"));
            config.define(
                "Vulkan_LIBRARY",
                molten_vk
                    .join("dylib")
                    .join("macOS")
                    .join("libMoltenVK.dylib"),
            );
        }
        config.build()
    };
    println!(
        "cargo:rustc-link-search=native={}",
        ncnn.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static:-bundle={}", "ncnn");
    println!("cargo:vulkan_dir={}", vulkan_dir.display());
    println!("cargo:vulkan_lib={}", vulkan_dir.join("lib").display());
    println!(
        "cargo:include={}",
        ncnn.join("include").join("ncnn").display()
    );
    println!("cargo:library={}", ncnn.join("lib").display());
}
