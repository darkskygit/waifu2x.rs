use cmake::Config;
use path_absolutize::Absolutize;
use std::env;
use std::fs::create_dir;
use std::path::PathBuf;

fn main() {
    let root_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let proj_dir = root_dir.absolutize().unwrap();
    let vulkan_dir = proj_dir.join("vulkan");
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
        create_dir(&ncnn_dir).unwrap_or_default();
        Config::new("ncnn")
            .generator(if cfg!(windows) {
                "NMake Makefiles"
            } else {
                "Unix Makefiles"
            })
            .out_dir(ncnn_dir)
            .static_crt(true)
            .profile("MinSizeRel")
            .env("VULKAN_SDK", &vulkan_dir)
            .define("MSVC_STATIC", "ON")
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
            .define("WITH_LAYER_HardSwish", "OFF")
            .build()
    };
    println!(
        "cargo:rustc-link-search=native={}",
        ncnn.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static-nobundle={}", "ncnn");
    println!("cargo:vulkan_dir={}", vulkan_dir.display());
    println!("cargo:vulkan_lib={}", vulkan_dir.join("lib").display());
    println!(
        "cargo:include={}",
        ncnn.join("include").join("ncnn").display()
    );
    println!("cargo:library={}", ncnn.join("lib").display());
}
