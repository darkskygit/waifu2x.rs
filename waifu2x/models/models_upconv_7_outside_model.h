#include "models-upconv_7_anime_style_art_rgb/noise0_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise1_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise2_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/scale2.0x_model.id.h"

extern "C" unsigned char *get_waifu2x_param(int noise, int scale, bool cunet);
extern "C" unsigned char *get_waifu2x_model(int noise, int scale, bool cunet);

namespace cunet {
	const unsigned char* get_param(int noise, int scale) {
		return get_waifu2x_param(noise, scale, true);
	}

	const unsigned char* get_model(int noise, int scale) {
		return get_waifu2x_model(noise, scale, true);
	}

	const int get_input(int noise, int scale) {
		switch (noise) {
		case 0:
			return noise0_scale2_0x_model_param_id::BLOB_Input1;
		case 1:
			return noise1_scale2_0x_model_param_id::BLOB_Input1;
		case 2:
			return noise2_scale2_0x_model_param_id::BLOB_Input1;
		case 3:
			return noise3_scale2_0x_model_param_id::BLOB_Input1;
		default:
			return scale2_0x_model_param_id::BLOB_Input1;
		}
	}

	const int get_extract(int noise, int scale) {
		switch (noise) {
		case 0:
			return noise0_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 1:
			return noise1_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 2:
			return noise2_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 3:
			return noise3_scale2_0x_model_param_id::BLOB_Eltwise4;
		default:
			return scale2_0x_model_param_id::BLOB_Eltwise4;
		}
	}

	int get_padding(int noise, int scale) {
		return 7;
	}
}

namespace upconv_7_anime_style_art_rgb {
	const unsigned char* get_param(int noise, int scale) {
		return get_waifu2x_param(noise, scale, false);
	}

	const unsigned char* get_model(int noise, int scale) {
		return get_waifu2x_model(noise, scale, false);
	}

	const int get_input(int noise, int scale) {
		switch (noise) {
		case 0:
			return noise0_scale2_0x_model_param_id::BLOB_Input1;
		case 1:
			return noise1_scale2_0x_model_param_id::BLOB_Input1;
		case 2:
			return noise2_scale2_0x_model_param_id::BLOB_Input1;
		case 3:
			return noise3_scale2_0x_model_param_id::BLOB_Input1;
		default:
			return scale2_0x_model_param_id::BLOB_Input1;
		}
	}

	const int get_extract(int noise, int scale) {
		switch (noise) {
		case 0:
			return noise0_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 1:
			return noise1_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 2:
			return noise2_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 3:
			return noise3_scale2_0x_model_param_id::BLOB_Eltwise4;
		default:
			return scale2_0x_model_param_id::BLOB_Eltwise4;
		}
	}

	int get_padding(int noise, int scale) {
		return 7;
	}
}