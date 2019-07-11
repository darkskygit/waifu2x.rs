#include "models-upconv_7_anime_style_art_rgb/noise0_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise0_scale2.0x_model.mem.h"
#include "models-upconv_7_anime_style_art_rgb/noise1_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise1_scale2.0x_model.mem.h"
#include "models-upconv_7_anime_style_art_rgb/noise2_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise2_scale2.0x_model.mem.h"
#include "models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.mem.h"
#include "models-upconv_7_anime_style_art_rgb/scale2.0x_model.id.h"
#include "models-upconv_7_anime_style_art_rgb/scale2.0x_model.mem.h"

namespace upconv_7_anime_style_art_rgb {

	const unsigned char* get_param(int noise, int scale) {
		switch (noise) {
		case 0:
			return &noise0_scale2_0x_model_param_bin[0];
		case 1:
			return &noise1_scale2_0x_model_param_bin[0];
		case 2:
			return &noise2_scale2_0x_model_param_bin[0];
		case 3:
			return &noise3_scale2_0x_model_param_bin[0];
		default:
			return &scale2_0x_model_param_bin[0];
		}
	}

	const unsigned char* get_model(int noise, int scale) {
		switch (noise) {
		case 0:
			return &noise0_scale2_0x_model_bin[0];
		case 1:
			return &noise1_scale2_0x_model_bin[0];
		case 2:
			return &noise2_scale2_0x_model_bin[0];
		case 3:
			return &noise3_scale2_0x_model_bin[0];
		default:
			return &scale2_0x_model_bin[0];
		}
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