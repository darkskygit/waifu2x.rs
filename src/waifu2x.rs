use libc::{c_int, c_void};

// for /f %f in ('dir /a/b *.prototxt') do @caffe2ncnn.exe %~nf.prototxt %~nf.json.caffemodel %~nf.param %~nf.bin 256 info.json
// for /f %f in ('dir /a/b *.param') do @ncnn2mem %~nf.param %~nf.bin %~nf.id.h %~nf.mem.h

#[repr(u8)]
enum Bool {
    False = 0,
    True = 1,
}

pub extern "C" {
    fn init_config(noise: c_int, scale: c_int, tilesize: c_int, is_cunet: Bool) -> *mut c_void;
    fn init_waifu2x(config: *mut c_void, gpuid: c_int) -> *mut c_void;
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