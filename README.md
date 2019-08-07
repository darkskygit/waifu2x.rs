# waifu2x.rs

A [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) Rust binding.

This library needs to be compiled with the rust nightly.

# Usage

```rust
use image::open;
use quicli::prelude::*;
use waifu2x::Waifu2x;

fn main() -> CliResult {
  let processer = Waifu2x::new(0, 0, 2, 128, true);
  let image = open("image.png")?;
  processer.proc_image(image, false).save("output.png")?;
}
```
