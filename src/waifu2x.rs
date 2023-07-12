use image::{imageops::CatmullRom, DynamicImage, RgbImage};
use libc::{c_int, c_void};

// for /f %f in ('dir /a/b *.prototxt') do @caffe2ncnn.exe %~nf.prototxt %~nf.json.caffemodel %~nf.param %~nf.bin 256 info.json
// for /f %f in ('dir /a/b *.param') do @ncnn2mem %~nf.param %~nf.bin %~nf.id.h %~nf.mem.h

#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum Waifu2xError {
    #[error("invalid noise level `{0}`; valid levels are -1,0,1,2,3")]
    InvalidNoise(i8),
    #[error("invalid scale `{0}`; valid scales are 1,2,4,8,16,32")]
    InvalidScale(u8),
    #[error("invalid gpuid `{0}`")]
    InvalidGpuId(u8),
}

#[repr(u8)]
enum Bool {
    False = 0,
    True = 1,
}

extern "C" {
    fn init_ncnn();
    fn init_config(noise: c_int, scale: c_int, tilesize: c_int, is_cunet: Bool) -> *mut c_void;
    fn init_waifu2x(config: *mut c_void, gpuid: c_int) -> *mut c_void;
    fn get_gpu_count() -> c_int;
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

#[derive(Debug)]
pub struct Waifu2x {
    config: *mut c_void,
    waifu2x: *mut c_void,
    scale: u8,
    runs: u8,
}

unsafe impl Send for Waifu2x {}

impl Waifu2x {
    pub fn init() {
        unsafe { init_ncnn() }
    }

    pub fn new(
        gpuid: u8,
        noise: i8,
        scale: u8,
        tilesize: u16,
        is_cunet: bool,
    ) -> Result<Self, Waifu2xError> {
        // valid noise values: -1,0,1,2,3
        if !(-1..=3).contains(&noise) {
            return Err(Waifu2xError::InvalidNoise(noise));
        }

        // check if scale is 0, is not power of 2, or more than 32
        // valid values are: 1,2,4,8,16,32
        if scale == 0 || scale & (scale - 1) != 0 || scale > 32 {
            return Err(Waifu2xError::InvalidScale(scale));
        }

        // check whether gpuid is valid or not
        let gpu_count = Self::get_gpu_count();
        // valid values: 0..gpu_count
        if !(0..gpu_count).contains(&gpuid.into()) {
            return Err(Waifu2xError::InvalidGpuId(gpuid));
        }

        unsafe {
            let config = init_config(
                i32::from(noise),
                // this always scales by 2 at a time
                2,
                i32::from(tilesize),
                if is_cunet { Bool::True } else { Bool::False },
            );

            let waifu2x = init_waifu2x(config, i32::from(gpuid));

            Ok(Self {
                config,
                waifu2x,
                scale,
                // calculate how many process image runs we need to get to the scale
                // every power of 2 yields +1
                // 1x = 0, 2x = 1, 4x = 2, 8x = 3, 16x = 4, 32x = 5
                runs: scale.ilog2() as u8,
            })
        }
    }

    pub fn get_gpu_count() -> i32 {
        unsafe { get_gpu_count() }
    }

    pub fn proc_image(&self, image: DynamicImage, downsampling: bool) -> DynamicImage {
        let width = image.width();
        let height = image.height();

        let mut image = self.proc_image_iter(image);

        if self.scale == 1 {
            // run scaling once and resize back to original size
            image = image.resize(width, height, CatmullRom);
        } else {
            for _ in 0..self.runs.saturating_sub(1) {
                image = self.proc_image_iter(image);
            }

            if downsampling {
                image = image.resize(width, height, CatmullRom);
            }
        }

        image
    }

    fn proc_image_iter(&self, image: DynamicImage) -> DynamicImage {
        let image_ptr = std::ptr::null_mut();
        let mut image_raw = image.to_rgb8().into_raw();
        let data = unsafe {
            proc_image(
                self.config,
                self.waifu2x,
                image_raw.as_mut_ptr() as *mut c_void,
                image.width() as i32,
                image.height() as i32,
                3,
                &image_ptr,
            )
        };

        let image = if let Some(new_image) = RgbImage::from_raw(
            image.width() * 2,
            image.height() * 2,
            unsafe {
                std::slice::from_raw_parts(
                    data as *const u8,
                    (image.width() * 2 * image.height() * 2 * 3) as usize,
                )
            }
            .to_vec(),
        ) {
            DynamicImage::ImageRgb8(new_image)
        } else {
            DynamicImage::ImageRgb8(
                RgbImage::from_raw(image.width(), image.height(), image_raw).unwrap(),
            )
        };

        unsafe {
            free_image(image_ptr);
        }

        image
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
