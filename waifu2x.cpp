// waifu2x implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <vector>

#ifdef WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // WIN32

using namespace std;

// ncnn
#include "layer_type.h"
#include "net.h"
#include "gpu.h"

static const uint32_t waifu2x_preproc_spv_data[] = {
	#include "waifu2x_preproc.spv.hex.h"
};
static const uint32_t waifu2x_preproc_fp16s_spv_data[] = {
	#include "waifu2x_preproc_fp16s.spv.hex.h"
};
#if !defined(NO_INT8_SUPPORT)
static const uint32_t waifu2x_preproc_int8s_spv_data[] = {
	#include "waifu2x_preproc_int8s.spv.hex.h"
};
#endif
static const uint32_t waifu2x_postproc_spv_data[] = {
	#include "waifu2x_postproc.spv.hex.h"
};
static const uint32_t waifu2x_postproc_fp16s_spv_data[] = {
	#include "waifu2x_postproc_fp16s.spv.hex.h"
};
#if !defined(NO_INT8_SUPPORT)
static const uint32_t waifu2x_postproc_int8s_spv_data[] = {
	#include "waifu2x_postproc_int8s.spv.hex.h"
};
#endif

#include "models/models.h"

class waifu2x_config {
public:
	int noise = 0;
	int scale = 2;
	int tilesize = 128;
	bool cunet = true;
public:
	waifu2x_config(int noise = 0, int scale = 2, int tilesize = 400, bool cunet = true) {
		if (noise >= 0) {
			this->noise = noise;
		}
		if (scale >= 1) {
			this->scale = scale;
		}
		if (tilesize >= 32) {
			this->tilesize = tilesize;
		}
		this->cunet = cunet;
	}
	const unsigned char* read_param() {
		return (this->cunet ? cunet::get_param : upconv_7_anime_style_art_rgb::get_param)(this->noise, this->scale);
	}
	const unsigned char* read_model() {
		return (this->cunet ? cunet::get_model : upconv_7_anime_style_art_rgb::get_model)(this->noise, this->scale);
	}
	const int input_blob() {
		return (this->cunet ? cunet::get_input : upconv_7_anime_style_art_rgb::get_input)(this->noise, this->scale);
	}
	const int extract_blob() {
		return (this->cunet ? cunet::get_extract : upconv_7_anime_style_art_rgb::get_extract)(this->noise, this->scale);
	}
	const int prepadding() {
		return (this->cunet ? cunet::get_padding : upconv_7_anime_style_art_rgb::get_padding)(this->noise, this->scale);
	}
};

class waifu2x_image {
public:
	int prepadding, scale = 2;
	int w, h, c;
	int prepadding_bottom, prepadding_right;
	int xtiles, ytiles;
	const int TILE_SIZE_X, TILE_SIZE_Y;
	unsigned char* data;
	ncnn::Mat buffer;
	waifu2x_image(waifu2x_config* config = new waifu2x_config())
		: prepadding(config->prepadding()), scale(config->scale), TILE_SIZE_X(config->tilesize), TILE_SIZE_Y(config->tilesize),
		w(0), h(0), c(0), prepadding_bottom(0), prepadding_right(0), xtiles(0), ytiles(0), data(0)
	{}
	waifu2x_image(int prepadding, int scale, int tilesize)
		:prepadding(prepadding), scale(scale), TILE_SIZE_X(tilesize), TILE_SIZE_Y(tilesize),
		w(0), h(0), c(0), prepadding_bottom(0), prepadding_right(0), xtiles(0), ytiles(0), data(0)
	{}
	~waifu2x_image() {
#ifdef WIN32
		free(this->data);
#else
		stbi_image_free(this->data);
#endif
	}
#ifdef WIN32
	void encode(const wchar_t* output) {
#else
	void encode(const char* output) {
#endif
#ifdef WIN32
		int ret = wic_encode_image(output, this->buffer.w, this->buffer.h, 3, this->buffer.data);
#else
		int ret = stbi_write_png(output, this->buffer.w, outrgb.h, 3, outrgb.data, 0);
#endif
		if (ret == 0)
		{
			fprintf(stderr, "encode image %ls failed\n", output);
			return;
		}
	}
#ifdef WIN32
	void decode(const wchar_t* input) {
#else
	void decode(const char* input) {
#endif
#ifdef WIN32
		this->data = wic_decode_image(input, &this->w, &this->h, &this->c);
#else
		this->data = stbi_load(input, &this->w, &this->h, &this->c, 3);
#endif 
		if (!this->data)
		{
			fprintf(stderr, "decode image %ls failed\n", input);
			return;
		}
		this->calc_config();
		this->buffer = ncnn::Mat(this->w * this->scale, this->h * this->scale, (size_t)3u, 3);
	}

private:
	void calc_config() {
		// prepadding
		int prepadding_bottom = this->prepadding;
		int prepadding_right = this->prepadding;
		if (this->scale == 1)
		{
			prepadding_bottom += (this->h + 3) / 4 * 4 - this->h;
			prepadding_right += (this->w + 3) / 4 * 4 - this->w;
		}
		if (this->scale == 2)
		{
			prepadding_bottom += (this->h + 1) / 2 * 2 - this->h;
			prepadding_right += (this->w + 1) / 2 * 2 - this->w;
		}
		this->prepadding_bottom = prepadding_bottom;
		this->prepadding_right = prepadding_right;
		this->xtiles = (this->w + this->TILE_SIZE_X - 1) / this->TILE_SIZE_X;
		this->ytiles = (this->h + this->TILE_SIZE_Y - 1) / this->TILE_SIZE_Y;
	}
	};

class waifu2x {
private:
	ncnn::Net net;
	ncnn::VulkanDevice* vkdev;
	ncnn::Pipeline* preproc;
	ncnn::Pipeline* postproc;
	int input_blob = 0;
	int extract_blob = 0;

public:
	waifu2x(int gpuid = 0) :vkdev(0), preproc(0), postproc(0) {
		ncnn::create_gpu_instance();

		int gpu_count = ncnn::get_gpu_count();
		if (gpuid < 0 || gpuid >= gpu_count)
		{
			fprintf(stderr, "invalid gpu device");
			ncnn::destroy_gpu_instance();
			return;
		}
		this->vkdev = ncnn::get_gpu_device(gpuid);

		ncnn::VkAllocator* blob_vkallocator = this->vkdev->acquire_blob_allocator();
		ncnn::VkAllocator* staging_vkallocator = this->vkdev->acquire_staging_allocator();

		ncnn::Option opt;
		opt.use_vulkan_compute = true;
		opt.blob_vkallocator = blob_vkallocator;
		opt.workspace_vkallocator = blob_vkallocator;
		opt.staging_vkallocator = staging_vkallocator;
		opt.use_fp16_packed = true;
		opt.use_fp16_storage = true;
		opt.use_fp16_arithmetic = false;
#if !defined(NO_INT8_SUPPORT)
		opt.use_int8_storage = true;
#else
		opt.use_int8_storage = false;
#endif
		opt.use_int8_arithmetic = false;

		this->net.opt = opt;
		this->net.set_vulkan_device(this->vkdev);
		this->init_proc();
	}
	~waifu2x()
	{
		// cleanup preprocess and postprocess pipeline
		delete this->preproc;
		delete this->postproc;

		this->vkdev->reclaim_blob_allocator(this->net.opt.blob_vkallocator);
		this->vkdev->reclaim_staging_allocator(this->net.opt.staging_vkallocator);

		ncnn::destroy_gpu_instance();
	}
	void load_models(const unsigned char* param, const  unsigned char* model) {
		this->net.load_param(param);
		this->net.load_model(model);
	}
	void set_model_blob(const int input, const  int extract) {
		this->input_blob = input;
		this->extract_blob = extract;
	}
private:
	void init_proc() {
		// initialize preprocess and postprocess pipeline
		vector<ncnn::vk_specialization_type> specializations(1);
#ifdef WIN32
		specializations[0].i = 1;
#else
		specializations[0].i = 0;
#endif

		this->preproc = new ncnn::Pipeline(this->vkdev);
		this->preproc->set_optimal_local_size_xyz(32, 32, 3);
#if !defined(NO_INT8_SUPPORT)
		if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			this->preproc->create(waifu2x_preproc_int8s_spv_data, sizeof(waifu2x_preproc_int8s_spv_data), "waifu2x_preproc_int8s", specializations, 2, 9);
		else
#endif
			if (this->net.opt.use_fp16_storage)
				this->preproc->create(waifu2x_preproc_fp16s_spv_data, sizeof(waifu2x_preproc_fp16s_spv_data), "waifu2x_preproc_fp16s", specializations, 2, 9);
			else
				this->preproc->create(waifu2x_preproc_spv_data, sizeof(waifu2x_preproc_spv_data), "waifu2x_preproc", specializations, 2, 9);

		this->postproc = new ncnn::Pipeline(this->vkdev);
		this->postproc->set_optimal_local_size_xyz(32, 32, 3);
#if !defined(NO_INT8_SUPPORT)
		if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			this->postproc->create(waifu2x_postproc_int8s_spv_data, sizeof(waifu2x_postproc_int8s_spv_data), "waifu2x_postproc_int8s", specializations, 2, 8);
		else
#endif
			if (this->net.opt.use_fp16_storage)
				this->postproc->create(waifu2x_postproc_fp16s_spv_data, sizeof(waifu2x_postproc_fp16s_spv_data), "waifu2x_postproc_fp16s", specializations, 2, 8);
			else
				this->postproc->create(waifu2x_postproc_spv_data, sizeof(waifu2x_postproc_spv_data), "waifu2x_postproc", specializations, 2, 8);
	}
public:
	void proc_image(waifu2x_image* image) {
		//#pragma omp parallel for num_threads(2)
		for (int yi = 0; yi < image->ytiles; yi++)
		{
			int in_tile_y0 = max(yi * image->TILE_SIZE_Y - image->prepadding, 0);
			int in_tile_y1 = min((yi + 1) * image->TILE_SIZE_Y + image->prepadding_bottom, image->h);

			ncnn::Mat in;
			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
				in = ncnn::Mat(image->w, (in_tile_y1 - in_tile_y0), image->data + in_tile_y0 * image->w * 3, (size_t)3u, 1);
			}
			else
			{
#ifdef WIN32
				in = ncnn::Mat::from_pixels(image->data + in_tile_y0 * image->w * 3, ncnn::Mat::PIXEL_BGR2RGB, image->w, (in_tile_y1 - in_tile_y0));
#else
				in = ncnn::Mat::from_pixels(image->data + in_tile_y0 * image->w * 3, ncnn::Mat::PIXEL_RGB, image->w, (in_tile_y1 - in_tile_y0));
#endif
			}

			ncnn::VkCompute cmd(this->vkdev);

			// upload
			ncnn::VkMat in_gpu;
			{
				in_gpu.create_like(in, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);

				in_gpu.prepare_staging_buffer();
				in_gpu.upload(in);

				cmd.record_upload(in_gpu);

				if (image->xtiles > 1)
				{
					cmd.submit_and_wait();
					cmd.reset();
				}
			}

			int out_tile_y0 = max(yi * image->TILE_SIZE_Y, 0);
			int out_tile_y1 = min((yi + 1) * image->TILE_SIZE_Y, image->h);

			ncnn::VkMat out_gpu;
			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
				out_gpu.create(image->w * image->scale, (out_tile_y1 - out_tile_y0) * image->scale, (size_t)3u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);
			}
			else
			{
				out_gpu.create(image->w * image->scale, (out_tile_y1 - out_tile_y0) * image->scale, 3, (size_t)4u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);
			}

			for (int xi = 0; xi < image->xtiles; xi++)
			{
				// preproc
				ncnn::VkMat in_tile_gpu;
				{
					// crop tile
					int tile_x0 = xi * image->TILE_SIZE_X;
					int tile_x1 = min((xi + 1) * image->TILE_SIZE_X, image->w) + image->prepadding + image->prepadding_right;
					int tile_y0 = yi * image->TILE_SIZE_Y;
					int tile_y1 = min((yi + 1) * image->TILE_SIZE_Y, image->h) + image->prepadding + image->prepadding_bottom;

					in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, (size_t)4u, 1, this->net.opt.blob_vkallocator, this->net.opt.staging_vkallocator);

					vector<ncnn::VkMat> bindings(2);
					bindings[0] = in_gpu;
					bindings[1] = in_tile_gpu;

					vector<ncnn::vk_constant_type> constants(9);
					constants[0].i = in_gpu.w;
					constants[1].i = in_gpu.h;
					constants[2].i = in_gpu.cstep;
					constants[3].i = in_tile_gpu.w;
					constants[4].i = in_tile_gpu.h;
					constants[5].i = in_tile_gpu.cstep;
					constants[6].i = max(image->prepadding - yi * image->TILE_SIZE_Y, 0);
					constants[7].i = image->prepadding;
					constants[8].i = xi * image->TILE_SIZE_X;

					cmd.record_pipeline(this->preproc, bindings, constants, in_tile_gpu);
				}

				// waifu2x
				ncnn::VkMat out_tile_gpu;
				{
					ncnn::Extractor ex = this->net.create_extractor();
					ex.input(this->input_blob, in_tile_gpu);

					ex.extract(this->extract_blob, out_tile_gpu, cmd);
				}

				// postproc
				{
					vector<ncnn::VkMat> bindings(2);
					bindings[0] = out_tile_gpu;
					bindings[1] = out_gpu;

					vector<ncnn::vk_constant_type> constants(8);
					constants[0].i = out_tile_gpu.w;
					constants[1].i = out_tile_gpu.h;
					constants[2].i = out_tile_gpu.cstep;
					constants[3].i = out_gpu.w;
					constants[4].i = out_gpu.h;
					constants[5].i = out_gpu.cstep;
					constants[6].i = xi * image->TILE_SIZE_X * image->scale;
					constants[7].i = out_gpu.w - xi * image->TILE_SIZE_X * image->scale;

					ncnn::VkMat dispatcher;
					dispatcher.w = out_gpu.w - xi * image->TILE_SIZE_X * image->scale;
					dispatcher.h = out_gpu.h;
					dispatcher.c = 3;

					cmd.record_pipeline(this->postproc, bindings, constants, dispatcher);
				}

				if (image->xtiles > 1)
				{
					cmd.submit_and_wait();
					cmd.reset();
				}
			}

			// download
			{
				out_gpu.prepare_staging_buffer();
				cmd.record_download(out_gpu);

				cmd.submit_and_wait();
			}

			if (this->net.opt.use_fp16_storage && this->net.opt.use_int8_storage)
			{
				ncnn::Mat out(out_gpu.w, out_gpu.h, (unsigned char*)image->buffer.data + yi * image->scale * image->TILE_SIZE_Y * image->w * image->scale * 3, (size_t)3u, 1);
				out_gpu.download(out);
			}
			else
			{
				ncnn::Mat out;
				out.create_like(out_gpu, this->net.opt.blob_allocator);
				out_gpu.download(out);
#ifdef WIN32
				out.to_pixels((unsigned char*)image->buffer.data + yi * image->scale * image->TILE_SIZE_Y * image->w * image->scale * 3, ncnn::Mat::PIXEL_RGB2BGR);
#else
				out.to_pixels((unsigned char*)image->buffer.data + yi * image->scale * image->TILE_SIZE_Y * image->w * image->scale * 3, ncnn::Mat::PIXEL_RGB);
#endif
			}
		}
	}
};

#ifdef WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
	if (optind >= argc || argv[optind][0] != L'-')
		return -1;

	wchar_t opt = argv[optind][1];
	const wchar_t* p = wcschr(optstring, opt);
	if (p == NULL)
		return L'?';

	optarg = NULL;

	if (p[1] == L':')
	{
		optind++;
		if (optind >= argc)
			return L'?';

		optarg = argv[optind];
	}

	optind++;

	return opt;
}
#else // WIN32
#include <unistd.h> // getopt()
#endif // WIN32

static void print_usage()
{
	fprintf(stderr, "Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...\n\n");
	fprintf(stderr, "  -h               show this help\n");
	fprintf(stderr, "  -i input-image   input image path (jpg/png)\n");
	fprintf(stderr, "  -o output-image  output image path (png)\n");
	fprintf(stderr, "  -n noise-level   denoise level (-1/0/1/2/3, default=0)\n");
	fprintf(stderr, "  -s scale         upscale ratio (1/2, default=2)\n");
	fprintf(stderr, "  -t tile-size     tile size (>=32, default=400)\n");
	fprintf(stderr, "  -m model-path    waifu2x model path (default=models-cunet)\n");
	fprintf(stderr, "  -g gpu-id        gpu device to use (default=0)\n");
}

#ifdef WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
#ifdef WIN32
	const wchar_t* imagepath = 0;
	const wchar_t* outputpngpath = 0;
	int noise = 0;
	int scale = 2;
	int tilesize = 400;
	int gpuid = 0;

	wchar_t opt;
	while ((opt = getopt(argc, argv, L"i:o:n:s:t:g:h")) != (wchar_t)-1)
	{
		switch (opt)
		{
		case L'i':
			imagepath = optarg;
			break;
		case L'o':
			outputpngpath = optarg;
			break;
		case L'n':
			noise = _wtoi(optarg);
			break;
		case L's':
			scale = _wtoi(optarg);
			break;
		case L't':
			tilesize = _wtoi(optarg);
			break;
		case L'g':
			gpuid = _wtoi(optarg);
			break;
		case L'h':
		default:
			print_usage();
			return -1;
		}
	}
#else // WIN32
	const char* imagepath = 0;
	const char* outputpngpath = 0;
	int noise = 0;
	int scale = 2;
	int tilesize = 400;
	int gpuid = 0;

	int opt;
	while ((opt = getopt(argc, argv, "i:o:n:s:t:g:h")) != -1)
	{
		switch (opt)
		{
		case 'i':
			imagepath = optarg;
			break;
		case 'o':
			outputpngpath = optarg;
			break;
		case 'n':
			noise = atoi(optarg);
			break;
		case 's':
			scale = atoi(optarg);
			break;
		case 't':
			tilesize = atoi(optarg);
			break;
		case 'g':
			gpuid = atoi(optarg);
			break;
		case 'h':
		default:
			print_usage();
			return -1;
		}
	}
#endif // WIN32

	if (!imagepath || !outputpngpath)
	{
		print_usage();
		return -1;
	}

	if (noise < -1 || noise > 3 || scale < 1 || scale > 2)
	{
		fprintf(stderr, "invalid noise or scale argument\n");
		return -1;
	}

	if (tilesize < 32)
	{
		fprintf(stderr, "invalid tilesize argument\n");
		return -1;
	}

#ifdef WIN32
	CoInitialize(0);
#endif
	auto config = waifu2x_config(noise, scale, tilesize, true);
	auto image = new waifu2x_image(&config);
	auto processer = new waifu2x(gpuid);
	processer->load_models(config.read_param(), config.read_model());
	processer->set_model_blob(config.input_blob(), config.extract_blob());
	image->decode(imagepath);
	processer->proc_image(image);
	image->encode(outputpngpath);
	delete processer;
	return 0;
}
