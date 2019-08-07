use cmake::Config;
use std::env;
use std::fs::create_dir;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let waifu2x_dir = out_dir.join("waifu2x");
    create_dir(&waifu2x_dir).unwrap_or_default();
    let waifu2x = Config::new("waifu2x")
        .out_dir(waifu2x_dir)
        .static_crt(true)
        .profile("MinSizeRel")
        .env("VULKAN_SDK", env::var("DEP_NCNN_VULKAN_DIR").unwrap())
        .define("MSVC_STATIC", "ON")
        .define("INCLUDE_LIST", env::var("DEP_NCNN_INCLUDE").unwrap())
        .define("LINK_LIST", env::var("DEP_NCNN_LIBRARY").unwrap())
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
    println!("cargo:rustc-link-lib=static-nobundle={}", "waifu2x-ncnn-vulkan");
    println!(
        "cargo:rustc-link-search=native={}",
        env::var("DEP_NCNN_LIBRARY").unwrap()
    );
    println!("cargo:rustc-link-lib=static-nobundle={}", "ncnn");
    println!(
        "cargo:rustc-link-search=native={}",
        env::var("DEP_NCNN_VULKAN_LIB").unwrap()
    );
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=static={}", "vulkan-1");
    } else {
        println!("cargo:rustc-link-lib=static={}", "gomp");
        println!("cargo:rustc-link-lib=static={}", "stdc++");
        println!("cargo:rustc-link-lib=dylib={}", "vulkan");
    }
}