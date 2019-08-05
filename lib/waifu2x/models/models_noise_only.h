#pragma once

#include "models-cunet/noise0_model.id.h"
#include "models-cunet/noise0_model.mem.h"
#include "models-cunet/noise1_model.id.h"
#include "models-cunet/noise1_model.mem.h"
#include "models-cunet/noise2_model.id.h"
#include "models-cunet/noise2_model.mem.h"
#include "models-cunet/noise3_model.id.h"
#include "models-cunet/noise3_model.mem.h"

namespace cunet {
	const unsigned char* get_param(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return &noise0_model_param_bin[0];
			case 1:
				return &noise1_model_param_bin[0];
			case 2:
				return &noise2_model_param_bin[0];
			case 3:
				return &noise3_model_param_bin[0];
			}
		default:
			return &noise0_model_param_bin[0];
		}
	}

	const unsigned char* get_model(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return &noise0_model_bin[0];
			case 1:
				return &noise1_model_bin[0];
			case 2:
				return &noise2_model_bin[0];
			case 3:
				return &noise3_model_bin[0];
			}
		default:
			return &noise0_model_bin[0];
		}
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
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return &noise0_model_param_bin[0];
			case 1:
				return &noise1_model_param_bin[0];
			case 2:
				return &noise2_model_param_bin[0];
			case 3:
				return &noise3_model_param_bin[0];
			}
		default:
			return &noise0_model_param_bin[0];
		}
	}

	const unsigned char* get_model(int noise, int scale) {
		switch (scale) {
		case 1:
			switch (noise) {
			case 0:
				return &noise0_model_bin[0];
			case 1:
				return &noise1_model_bin[0];
			case 2:
				return &noise2_model_bin[0];
			case 3:
				return &noise3_model_bin[0];
			}
		default:
			return &noise0_model_bin[0];
		}
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