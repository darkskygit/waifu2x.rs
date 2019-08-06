use cmake::Config;
use path_absolutize::Absolutize;
use std::env;
use std::fs::create_dir;
use std::path::PathBuf;

fn main() {
    let proj_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .absolutize()
        .unwrap();
    let vulkan_dir = proj_dir.join("lib").join("vulkan");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ncnn_dir = out_dir.join("ncnn");
    create_dir(&ncnn_dir).unwrap_or_default();
    let ncnn = Config::new("lib/ncnn")
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
    let waifu2x_dir = out_dir.join("waifu2x");
    create_dir(&waifu2x_dir).unwrap_or_default();
    let waifu2x = Config::new("lib/waifu2x")
        .out_dir(waifu2x_dir)
        .static_crt(true)
        .profile("MinSizeRel")
        .env("VULKAN_SDK", &vulkan_dir)
        .define("MSVC_STATIC", "ON")
        .define("INCLUDE_LIST", ncnn.join("include").join("ncnn"))
        .define("LINK_LIST", ncnn.join("lib"))
        .define(
            if cfg!(feature = "upconv7") {
                "WAIFU2X_UPCONV7_ONLY"
            } else if cfg!(feature = "noise") {
                "WAIFU2X_NOISE_ONLY"
            } else {
                "WAIFU2X_FULL"
            },
            "ON",
        )
        .build();
    println!(
        "cargo:rustc-link-search=native={}",
        waifu2x.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static={}", "waifu2x-ncnn-vulkan");
    println!(
        "cargo:rustc-link-search=native={}",
        vulkan_dir.join("lib").display()
    );
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=static={}", "vulkan-1");
    } else {
        println!("cargo:rustc-link-lib=static={}", "gomp");
        println!("cargo:rustc-link-lib=static={}", "stdc++");
        println!("cargo:rustc-link-lib=dylib={}", "vulkan");
    }
}