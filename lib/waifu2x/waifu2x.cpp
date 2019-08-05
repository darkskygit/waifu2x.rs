// waifu2x implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <vector>

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

#if defined(WAIFU2X_NOISE_ONLY)
#include "models/models_noise_only.h"
#elif defined(WAIFU2X_UPCONV7_ONLY)
#include "models/models_upconv_7_only.h"
#else
#include "models/models.h"
#endif

class waifu2x_config
{
public:
	int noise = 0;
	int scale = 2;
	int tilesize = 128;
	bool cunet = true;
	waifu2x_config(int noise = 0, int scale = 2, int tilesize = 400, bool cunet = true)
	{
		if (noise >= 0)
		{
			this->noise = noise;
		}
		if (scale >= 1)
		{
			this->scale = scale;
		}
		if (tilesize >= 32)
		{
			this->tilesize = tilesize;
		}
		this->cunet = cunet;
	}
	const unsigned char* read_param()
	{
		return (this->cunet ? cunet::get_param : upconv_7_anime_style_art_rgb::get_param)(this->noise, this->scale);
	}
	const unsigned char* read_model()
	{
		return (this->cunet ? cunet::get_model : upconv_7_anime_style_art_rgb::get_model)(this->noise, this->scale);
	}
	const int input_blob()
	{
		return (this->cunet ? cunet::get_input : upconv_7_anime_style_art_rgb::get_input)(this->noise, this->scale);
	}
	const int extract_blob()
	{
		return (this->cunet ? cunet::get_extract : upconv_7_anime_style_art_rgb::get_extract)(this->noise, this->scale);
	}
	const int prepadding()
	{
		return (this->cunet ? cunet::get_padding : upconv_7_anime_style_art_rgb::get_padding)(this->noise, this->scale);
	}
};

class waifu2x_image
{
public:
	int prepadding, scale;
	int w, h, c;
	int prepadding_bottom, prepadding_right;
	int xtiles, ytiles;
	const int TILE_SIZE_X, TILE_SIZE_Y;
	unsigned char* data;
	ncnn::Mat buffer;
	waifu2x_image(waifu2x_config* config = new waifu2x_config())
		: prepadding(config->prepadding()), scale(config->scale), TILE_SIZE_X(config->tilesize), TILE_SIZE_Y(config->tilesize),
		w(0), h(0), c(0), prepadding_bottom(0), prepadding_right(0), xtiles(0), ytiles(0), data(0)
	{
	}
	void decode(unsigned char* data, int w, int h, int c)
	{
		this->data = data;
		this->w = w;
		this->h = h;
		this->c = c;
		this->calc_config();
		this->buffer = ncnn::Mat(this->w * this->scale, this->h * this->scale, (size_t)3u, 3);
	}

private:
	void calc_config()
	{
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

class waifu2x
{
private:
	ncnn::Net net;
	ncnn::VulkanDevice* vkdev;
	ncnn::Pipeline* preproc;
	ncnn::Pipeline* postproc;
	int input_blob = 0;
	int extract_blob = 0;

public:
	waifu2x(int gpuid = 0) : vkdev(0), preproc(0), postproc(0)
	{
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
	void load_models(const unsigned char* param, const unsigned char* model)
	{
		this->net.load_param(param);
		this->net.load_model(model);
	}
	void set_model_blob(const int input, const int extract)
	{
		this->input_blob = input;
		this->extract_blob = extract;
	}

private:
	void init_proc()
	{
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
	void proc_image(waifu2x_image* image)
	{
		#pragma omp parallel for num_threads(this->vkdev->info.compute_queue_count)
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
				in = ncnn::Mat::from_pixels(image->data + in_tile_y0 * image->w * 3, ncnn::Mat::PIXEL_RGB, image->w, (in_tile_y1 - in_tile_y0));
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
				out.to_pixels((unsigned char*)image->buffer.data + yi * image->scale * image->TILE_SIZE_Y * image->w * image->scale * 3, ncnn::Mat::PIXEL_RGB);
			}
		}
	}
};

extern "C" waifu2x_config* init_config(int noise, int scale, int tilesize, bool is_cunet)
{
	return new waifu2x_config(noise, scale, tilesize, is_cunet);
}

extern "C" waifu2x* init_waifu2x(waifu2x_config* config, int gpuid)
{
	auto processer = new waifu2x(gpuid);
	processer->load_models(config->read_param(), config->read_model());
	processer->set_model_blob(config->input_blob(), config->extract_blob());
	return processer;
}

extern "C" void* proc_image(waifu2x_config* config, waifu2x* processer, unsigned char* data, int w, int h, int c, waifu2x_image*& image)
{
	image = new waifu2x_image(config);
	image->decode(data, w, h, c);
	processer->proc_image(image);
	return image->buffer.data;
}

extern "C" void free_image(waifu2x_image* image)
{
	if (image)
		delete image;
}

extern "C" void free_waifu2x(waifu2x_config* config, waifu2x* processer)
{
	if (config)
		delete config;
	if (processer)
		delete processer;
}
