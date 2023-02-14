use image::{imageops::CatmullRom, DynamicImage, RgbImage};
use libc::{c_int, c_void};

// for /f %f in ('dir /a/b *.prototxt') do @caffe2ncnn.exe %~nf.prototxt %~nf.json.caffemodel %~nf.param %~nf.bin 256 info.json
// for /f %f in ('dir /a/b *.param') do @ncnn2mem %~nf.param %~nf.bin %~nf.id.h %~nf.mem.h

#[repr(u8)]
enum Bool {
    False = 0,
    True = 1,
}

extern "C" {
    fn init_ncnn();
    fn init_config(noise: c_int, scale: c_int, tilesize: c_int, is_cunet: Bool) -> *mut c_void;
    fn init_waifu2x(config: *mut c_void, gpuid: c_int) -> *mut c_void;
    fn get_gpu_count(processer: *mut c_void) -> c_int;
    fn proc_image(
        config: *mut c_void,
        processer: *mut c_void,
        data: *mut c_void,
        w: c_int,
        h: c_int,
        c: c_int,
        image: &*mut c_void,
    ) -> *mut c_void;
    fn free_image(image: *mut c_void);
    fn free_waifu2x(config: *mut c_void, processer: *mut c_void);
}

pub struct Waifu2x {
    config: *mut c_void,
    waifu2x: *mut c_void,
    scale: u8,
}

unsafe impl Send for Waifu2x {}

impl Waifu2x {
    pub fn init() {
        unsafe { init_ncnn() }
    }
    pub fn new(gpuid: u8, noise: u8, scale: u8, tilesize: u16, is_cunet: bool) -> Self {
        unsafe {
            let config = init_config(
                i32::from(noise),
                i32::from(scale),
                i32::from(tilesize),
                if is_cunet { Bool::True } else { Bool::False },
            );
            let waifu2x = init_waifu2x(config, i32::from(gpuid));
            Self {
                config,
                waifu2x,
                scale,
            }
        }
    }
    pub fn get_gpu_count(&self) -> i32 {
        unsafe { get_gpu_count(self.waifu2x) }
    }
    pub fn proc_image(&self, image: DynamicImage, downsampling: bool) -> DynamicImage {
        let image_ptr = std::ptr::null_mut();
        let mut image_raw = image.to_rgb8().into_raw();
        unsafe {
            let data = proc_image(
                self.config,
                self.waifu2x,
                image_raw.as_mut_ptr() as *mut c_void,
                image.width() as i32,
                image.height() as i32,
                3,
                &image_ptr,
            );
            let image = if let Some(new_image) = RgbImage::from_raw(
                image.width() * u32::from(self.scale),
                image.height() * u32::from(self.scale),
                std::slice::from_raw_parts(
                    data as *const u8,
                    (image.width()
                        * u32::from(self.scale)
                        * image.height()
                        * u32::from(self.scale)
                        * 3) as usize,
                )
                .to_vec(),
            ) {
                let new_image = DynamicImage::ImageRgb8(new_image);
                if downsampling && self.scale > 1 {
                    new_image.resize(image.width(), image.height(), CatmullRom)
                } else {
                    new_image
                }
            } else {
                DynamicImage::ImageRgb8(
                    RgbImage::from_raw(image.width(), image.height(), image_raw).unwrap(),
                )
            };
            free_image(image_ptr);
            image
        }
    }
}

impl Drop for Waifu2x {
    fn drop(&mut self) {
        unsafe {
            free_waifu2x(self.config, self.waifu2x);
        }
    }
}

#[no_mangle]
#[cfg(all(feature = "upconv7_outside_model", feature = "model_bundled"))]
extern "C" fn get_waifu2x_param(noise: c_int, _scale: c_int, _cunet: Bool) -> *const u8 {
    match noise {
        0 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise0_scale2.0x_model.param.bin"
        )
        .as_ptr(),
        1 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise1_scale2.0x_model.param.bin"
        )
        .as_ptr(),
        2 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise2_scale2.0x_model.param.bin"
        )
        .as_ptr(),
        3 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.param.bin"
        )
        .as_ptr(),
        _ => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/scale2.0x_model.param.bin"
        )
        .as_ptr(),
    }
}

#[no_mangle]
#[cfg(all(feature = "upconv7_outside_model", feature = "model_bundled"))]
extern "C" fn get_waifu2x_model(noise: c_int, _scale: c_int, _cunet: Bool) -> *const u8 {
    match noise {
        0 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise0_scale2.0x_model.bin"
        )
        .as_ptr(),
        1 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise1_scale2.0x_model.bin"
        )
        .as_ptr(),
        2 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise2_scale2.0x_model.bin"
        )
        .as_ptr(),
        3 => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.bin"
        )
        .as_ptr(),
        _ => include_bytes!(
            "../waifu2x/models/models-upconv_7_anime_style_art_rgb/scale2.0x_model.bin"
        )
        .as_ptr(),
    }
}

#[no_mangle]
#[cfg(all(feature = "noise_outside_model", feature = "model_bundled"))]
pub extern "C" fn get_waifu2x_param(noise: c_int, scale: c_int, _cunet: Bool) -> *const u8 {
    match scale {
        1 => match noise {
            0 => include_bytes!("../waifu2x/models/models-cunet/noise0_model.param.bin").as_ptr(),
            1 => include_bytes!("../waifu2x/models/models-cunet/noise1_model.param.bin").as_ptr(),
            2 => include_bytes!("../waifu2x/models/models-cunet/noise2_model.param.bin").as_ptr(),
            3 => include_bytes!("../waifu2x/models/models-cunet/noise3_model.param.bin").as_ptr(),
        },
        _ => include_bytes!("../waifu2x/models/models-cunet/noise0_model.param.bin").as_ptr(),
    }
}

#[no_mangle]
#[cfg(all(feature = "noise_outside_model", feature = "model_bundled"))]
pub extern "C" fn get_waifu2x_model(noise: c_int, scale: c_int, _cunet: Bool) -> *const u8 {
    match scale {
        1 => match noise {
            0 => include_bytes!("../waifu2x/models/models-cunet/noise0_model.bin").as_ptr(),
            1 => include_bytes!("../waifu2x/models/models-cunet/noise1_model.bin").as_ptr(),
            2 => include_bytes!("../waifu2x/models/models-cunet/noise2_model.bin").as_ptr(),
            3 => include_bytes!("../waifu2x/models/models-cunet/noise3_model.bin").as_ptr(),
        },
        _ => include_bytes!("../waifu2x/models/models-cunet/noise0_model.bin").as_ptr(),
    }
}
