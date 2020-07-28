// model_execution.cpp : Inference example for Mut1ny's Face segmentation model
//                       using PyTorch CPP backend. You need PyTorch CPP build for this
//                       and the FreeImage Library (https://freeimage.sourceforge.io)

#include <iostream>
#include<iostream>
#include<memory>
#include<torch/script.h>
#include <vector>
#include <list>
#include <Freeimage.h>

const unsigned char LABELRED[] = {0,
                                  0,
	                              0,
	                              255,
	                              0,
	                              255,
	                              255,
	                              255,
	                              128,
	                              255,
	                              0};
const unsigned char LABELGREEN[] = {0,
                                    255,
	                                0,
	                                0,
	                                255,
	                                255,
	                                0,
	                                255,
	                                128,
	                                192,
	                                128};
const unsigned char LABELBLUE[] = {0,
                                   0,
	                               255,
	                               0,
	                               255,
	                               0,
	                               255,
	                               255,
	                               128,
	                               192,
	                               128};

const int NETWORKINPUTWIDTH = 344;
const int NETWORKINPUTHEIGHT = 344;

unsigned DLL_CALLCONV
myReadProc(void *buffer, unsigned size, unsigned count, fi_handle handle) {
	return fread(buffer, size, count, (FILE *)handle);
}

unsigned DLL_CALLCONV
myWriteProc(void *buffer, unsigned size, unsigned count, fi_handle handle) {
	return fwrite(buffer, size, count, (FILE *)handle);
}

int DLL_CALLCONV
mySeekProc(fi_handle handle, long offset, int origin) {
	return fseek((FILE *)handle, offset, origin);
}

long DLL_CALLCONV
myTellProc(fi_handle handle) {
	return ftell((FILE *)handle);
}

std::list<float> Softmax(float* input, int numElements)
{
	std::list<float> returnVals;
	float sum = 0.0f;
	for (int n = 0; n < numElements; n++)
	{
		float expon = expf(input[n]);
		returnVals.push_back(expon);
		sum += expon;
	}
	std::list<float>::iterator iter = returnVals.begin();
	for (int n = 0; n < numElements; n++, iter++)
		(*iter) /= sum;
	return returnVals;
}

void ProcessInputImage(FILE* fp, std::vector<float>& output, std::vector<int64_t>& inDims)
{
	FreeImageIO io;
	io.read_proc = myReadProc;
	io.write_proc = myWriteProc;
	io.seek_proc = mySeekProc;
	io.tell_proc = myTellProc;
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromHandle(&io, (fi_handle)fp, 0);
	bool resized = false;
	if (fif != FIF_UNKNOWN)
	{
		FIBITMAP *dib = FreeImage_LoadFromHandle(fif, &io, (fi_handle)fp, 0);
		unsigned int height = FreeImage_GetHeight(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		FIBITMAP* dibOut = dib;
		__int64 newWidth = width;
		__int64 newHeight = height;
		if (height > width)
		{
			newHeight = inDims[2];
			newWidth = (newHeight * width) / height;
		}
		else {
			newWidth = inDims[3];
			newHeight = (newWidth * height) / width;
		}

		newWidth = inDims[3];
		newHeight = inDims[2];
		float xScale = float(width) / float(newWidth);
		float yScale = float(height) / float(newHeight);
		__int64 outWidth = inDims[3];
		__int64 outHeight = inDims[2];
		output.resize(outWidth*outHeight * 3);
#pragma omp parallel for
		for (int n = 0; n < outWidth * outHeight * 3; n++)
			output[n] = 0.0f;
		__int64 deltaX = (outWidth - newWidth) >> 1;
		__int64 deltaY = (outHeight - newHeight) >> 1;

#pragma omp parallel for
		for (int y = 0; y < newHeight; y++)
		{
			// Y-Direction is flipped
			int srcY = height - 1 - (int)(y * xScale);
			BYTE* bits = FreeImage_GetScanLine(dib, srcY);

			for (int x = 0; x < newWidth; x++)
			{
				int srcX = int(x*xScale);
				float red = float(bits[srcX * (bpp == 32 ? 4 : 3) + 2]);
				red -= 128.0f;
				red /= 128.0f;
				float green = float(bits[srcX * (bpp == 32 ? 4 : 3) + 1]);
				green -= 128.0f;
				green /= 128.0f;
				float blue = float(bits[srcX * (bpp == 32 ? 4 : 3)]);

				blue -= 128.0f;
				blue /= 128.0f;
				output[((x + deltaX) + (y + deltaY) * outWidth)] = red;
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight] = green;
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight * 2] = blue;
			}
		}
		FreeImage_Unload(dib);
	}
}

int main()
{
	std::shared_ptr<torch::Device> m_pDevice;
	// if you have CUDA
	m_pDevice = std::shared_ptr<torch::Device>(new torch::Device(torch::kCUDA));
	// if you have NOT CUDA but CPU uncomment this line and comment line above
	// m_pDevice = std::shared_ptr<torch::Device>(new torch::Device(torch::kCPU));
	torch::jit::script::Module m_pModule;
	m_pModule = torch::jit::load("facesegmentation_344.pt", *m_pDevice);
	m_pModule.eval();
	FILE* fpInput = NULL;
	fopen_s(&fpInput, "D:\\work\\testimages\\final2\\synthesized0000.png", "rb");
	if (fpInput)
	{
		std::vector<float> tensorInput;
		tensorInput.resize(344 * 344 * 3);
		std::vector<int64_t> inDims;
		inDims.resize(4);
		inDims[0] = 1;
		inDims[1] = 3;
		inDims[2] = NETWORKINPUTHEIGHT;
		inDims[3] = NETWORKINPUTWIDTH;
		ProcessInputImage(fpInput, tensorInput, inDims);
		std::vector<torch::jit::IValue> inputs;
		auto testInput = torch::zeros({ 1,3,NETWORKINPUTHEIGHT,NETWORKINPUTWIDTH });
		float* ptr = (float*)testInput.data_ptr();
		memcpy(ptr, tensorInput.data(), sizeof(float)*NETWORKINPUTWIDTH*NETWORKINPUTHEIGHT * 3);
		auto gpu_rand = testInput.to(*m_pDevice);
		inputs.push_back(gpu_rand);
		//	inputs.push_back(testInput);
		auto temp = m_pModule.forward(inputs);
		at::Tensor output = temp.toTensor();
		at::Tensor cpu_result = output.cpu();
		float* outPtr = (float*)cpu_result.data_ptr();
		std::vector<int64_t> outDims;
		outDims.push_back(output.size(0));
		outDims.push_back(output.size(1));
		outDims.push_back(output.size(2));
		outDims.push_back(output.size(3));

		FIBITMAP* dibOut = FreeImage_Allocate(outDims[3], outDims[2], 24);
		fprintf(stderr, "OUTPUT = %d %d %d \n", outDims[1], outDims[2], outDims[3]);
#pragma omp parallel for
		for (int y = 0; y < outDims[2]; y++)
		{
			int srcY = outDims[2] - 1 - y;
			BYTE* bits = FreeImage_GetScanLine(dibOut, srcY);
			for (int x = 0; x < outDims[3]; x++)
			{
				std::vector<float> predVals;
				predVals.resize(outDims[1]);
				for (int n = 0; n < outDims[1]; n++)
				{
					predVals[n] = outPtr[x + y * outDims[3] + n * outDims[2] * outDims[3]];
				}
				std::list<float> output = Softmax(predVals.data(), outDims[1]);
				std::list<float> sorted;
				sorted.insert(sorted.begin(), output.begin(), output.end());
				sorted.sort();
				float maxVal = sorted.back();
				std::list<float>::const_iterator iter;
				int idx = 0;
				for (iter = output.begin(); iter != output.end(); iter++, idx++)
				{
					if (fabs(maxVal - *iter) < 1E-08)
					{
						break;
					}
				}
				bits[FI_RGBA_RED] = LABELRED[idx];
				bits[FI_RGBA_GREEN] = LABELGREEN[idx];
				bits[FI_RGBA_BLUE] = LABELBLUE[idx];
				bits += 3;
			}
		}
		FreeImage_Save(FIF_PNG, dibOut, "labels.png");
	}
	fclose(fpInput);
	return 0;
}

