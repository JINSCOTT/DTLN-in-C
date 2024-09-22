#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <stdio.h>
#include "dtln.hpp"
#include <complex>
#include <chrono>
#ifdef __EMSCRIPTEN__
void main_loop__em()
{
}
#endif

DTLN d;
int error = 0;
std::vector<float> output = std::vector<float>(512, 0);

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{

	auto start = std::chrono::steady_clock::now();
	std::vector<float> input((float*)pInput, (float*)pInput + 128);
	error = d.inference(input, output);
	memcpy((float*)pOutput, output.data(), 128 * sizeof(float));
	auto finish = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<
		std::chrono::duration<double>>(finish - start).count();
	std::cout << elapsed_seconds << "s\n";
}

int main(int argc, char** argv)
{
	FILE* stream;
	freopen_s(&stream, "output.txt", "w", stdout);

	ma_result result;
	ma_device_config deviceConfig;
	ma_device device;

	deviceConfig = ma_device_config_init(ma_device_type_duplex);
	deviceConfig.sampleRate = 16000;
	deviceConfig.periodSizeInFrames = 128;
	deviceConfig.capture.pDeviceID = NULL;
	deviceConfig.capture.format = ma_format_f32;
	deviceConfig.capture.channels = 1;
	deviceConfig.capture.shareMode = ma_share_mode_shared;
	deviceConfig.playback.pDeviceID = NULL;
	deviceConfig.playback.format = ma_format_f32;
	deviceConfig.playback.channels = 1;
	deviceConfig.dataCallback = data_callback;
	result = ma_device_init(NULL, &deviceConfig, &device);
	if (result != MA_SUCCESS) {
		std::cout << "init fail\n";
		return result;
	}

#ifdef __EMSCRIPTEN__
	getchar();
#endif

	ma_device_start(&device);

#ifdef __EMSCRIPTEN__
	emscripten_set_main_loop(main_loop__em, 0, 1);
#else
	printf("Press Enter to quit...\n");
	getchar();
#endif

	ma_device_uninit(&device);

	(void)argc;
	(void)argv;
	return 0;
}