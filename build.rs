use cmake::Config;
use path_absolutize::Absolutize;
use std::env;
use std::fs::create_dir;
use std::path::PathBuf;

fn main() {
    let proj_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .absolutize()
        .unwrap();
    let vulkan_dir = proj_dir.join("vulkan");
    println!("vulkan_dir: {}", vulkan_dir.display());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ncnn_dir = out_dir.join("ncnn");
    create_dir(&ncnn_dir).unwrap_or_default();
    let ncnn = Config::new("ncnn")
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
        .build();
    println!(
        "cargo:rustc-link-search=native={}",
        ncnn.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static={}", "ncnn");
    println!("cargo:vulkan_dir={}", vulkan_dir.display());
    println!("cargo:vulkan_lib={}", vulkan_dir.join("lib").display());
    println!("cargo:include={}", ncnn.join("include").join("ncnn").display());
    println!("cargo:library={}", ncnn.join("lib").display());
}