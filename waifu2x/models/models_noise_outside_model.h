#pragma once

#include "models-cunet/noise0_model.id.h"
#include "models-cunet/noise1_model.id.h"
#include "models-cunet/noise2_model.id.h"
#include "models-cunet/noise3_model.id.h"

extern "C" unsigned char *get_waifu2x_param(int noise, int scale, bool cunet);
extern "C" unsigned char *get_waifu2x_model(int noise, int scale, bool cunet);

namespace cunet {
	const unsigned char* get_param(int noise, int scale) {
		return get_waifu2x_param(noise, scale, false);
	}

	const unsigned char* get_model(int noise, int scale) {
		return get_waifu2x_model(noise, scale, true);
	}

	const int get_input(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return noise0_model_param_id::BLOB_Input1;
			case 1:
				return noise1_model_param_id::BLOB_Input1;
			case 2:
				return noise2_model_param_id::BLOB_Input1;
			case 3:
				return noise3_model_param_id::BLOB_Input1;
			}
		default:
			return noise0_model_param_id::BLOB_Input1;
		}
	}

	const int get_extract(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return noise0_model_param_id::BLOB_Eltwise4;
			case 1:
				return noise1_model_param_id::BLOB_Eltwise4;
			case 2:
				return noise2_model_param_id::BLOB_Eltwise4;
			case 3:
				return noise3_model_param_id::BLOB_Eltwise4;
			}
		default:
			return noise0_model_param_id::BLOB_Eltwise4;
		}
	}

	int get_padding(int noise, int scale) {
		return 28;
	}
}

namespace upconv_7_anime_style_art_rgb {
	const unsigned char* get_param(int noise, int scale) {
		return get_waifu2x_param(noise, scale, false);
	}

	const unsigned char* get_model(int noise, int scale) {
		return get_waifu2x_model(noise, scale, true);
	}

	const int get_input(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return noise0_model_param_id::BLOB_Input1;
			case 1:
				return noise1_model_param_id::BLOB_Input1;
			case 2:
				return noise2_model_param_id::BLOB_Input1;
			case 3:
				return noise3_model_param_id::BLOB_Input1;
			}
		default:
			return noise0_model_param_id::BLOB_Input1;
		}
	}

	const int get_extract(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return noise0_model_param_id::BLOB_Eltwise4;
			case 1:
				return noise1_model_param_id::BLOB_Eltwise4;
			case 2:
				return noise2_model_param_id::BLOB_Eltwise4;
			case 3:
				return noise3_model_param_id::BLOB_Eltwise4;
			}
		default:
			return noise0_model_param_id::BLOB_Eltwise4;
		}
	}

	int get_padding(int noise, int scale) {
		return 28;
	}
}