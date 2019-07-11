#pragma once
#include "models-cunet/noise0_model.id.h"
#include "models-cunet/noise0_model.mem.h"
#include "models-cunet/noise0_scale2.0x_model.id.h"
#include "models-cunet/noise0_scale2.0x_model.mem.h"
#include "models-cunet/noise1_model.id.h"
#include "models-cunet/noise1_model.mem.h"
#include "models-cunet/noise1_scale2.0x_model.id.h"
#include "models-cunet/noise1_scale2.0x_model.mem.h"
#include "models-cunet/noise2_model.id.h"
#include "models-cunet/noise2_model.mem.h"
#include "models-cunet/noise2_scale2.0x_model.id.h"
#include "models-cunet/noise2_scale2.0x_model.mem.h"
#include "models-cunet/noise3_model.id.h"
#include "models-cunet/noise3_model.mem.h"
#include "models-cunet/noise3_scale2.0x_model.id.h"
#include "models-cunet/noise3_scale2.0x_model.mem.h"
#include "models-cunet/scale2.0x_model.id.h"
#include "models-cunet/scale2.0x_model.mem.h"

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
	case 2:
		switch (noise) {
		case 0:
			return &noise0_scale2_0x_model_param_bin[0];
		case 1:
			return &noise1_scale2_0x_model_param_bin[0];
		case 2:
			return &noise2_scale2_0x_model_param_bin[0];
		case 3:
			return &noise3_scale2_0x_model_param_bin[0];
		}
	default:
		return &scale2_0x_model_param_bin[0];
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
	case 2:
		switch (noise) {
		case 0:
			return &noise0_scale2_0x_model_bin[0];
		case 1:
			return &noise1_scale2_0x_model_bin[0];
		case 2:
			return &noise2_scale2_0x_model_bin[0];
		case 3:
			return &noise3_scale2_0x_model_bin[0];
		}
	default:
		return &scale2_0x_model_bin[0];
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
	case 2:
		switch (noise) {
		case 0:
			return noise0_scale2_0x_model_param_id::BLOB_Input1;
		case 1:
			return noise1_scale2_0x_model_param_id::BLOB_Input1;
		case 2:
			return noise2_scale2_0x_model_param_id::BLOB_Input1;
		case 3:
			return noise3_scale2_0x_model_param_id::BLOB_Input1;
		}
	default:
		return scale2_0x_model_param_id::BLOB_Input1;
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
	case 2:
		switch (noise) {
		case 0:
			return noise0_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 1:
			return noise1_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 2:
			return noise2_scale2_0x_model_param_id::BLOB_Eltwise4;
		case 3:
			return noise3_scale2_0x_model_param_id::BLOB_Eltwise4;
		}
	default:
		return scale2_0x_model_param_id::BLOB_Eltwise4;
	}
}