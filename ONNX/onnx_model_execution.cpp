// ONNXTest.cpp

#include <stdio.h>
#include <tchar.h>
#include <string>
#include<vector>
#include<omp.h>
// FREEImage library
#include "Freeimage.h"
// ONNXRuntime
#include "onnxruntime_c_api.h"
// CUDA ONNXRuntime
#include "cuda_provider_factory.h"


unsigned DLL_CALLCONV
myReadProc(void *buffer, unsigned size, unsigned count, fi_handle handle) {
	return (unsigned)fread(buffer, size, count, (FILE *)handle);
}

unsigned DLL_CALLCONV
myWriteProc(void *buffer, unsigned size, unsigned count, fi_handle handle) {
	return (unsigned)fwrite(buffer, size, count, (FILE *)handle);
}

int DLL_CALLCONV
mySeekProc(fi_handle handle, long offset, int origin) {
	return fseek((FILE *)handle, offset, origin);
}

long DLL_CALLCONV
myTellProc(fi_handle handle) {
	return ftell((FILE *)handle);
}

#define max(a, b) (a > b ? a : b);

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);


void CheckStatus(OrtStatus* status)
{
	if (status != NULL) {
		const char* msg = g_ort->GetErrorMessage(status);
		fprintf(stderr, msg);
	}
}

float Sigmoid(float input)
{
	return 1.0f / (1 + expf(-input));
}

float* Softmax(float* input, int numElements)
{
	float* returnVals = new float[numElements];
	float sum = 0.0f;
	for (int n = 0; n < numElements; n++)
	{
		returnVals[n] = expf(input[n]);
		sum += returnVals[n];
	}
	for (int n = 0; n < numElements; n++)
		returnVals[n] /= sum;
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
		} else {
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
				float red = float(bits[srcX * (bpp == 32 ? 4 : 3) +2]);
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
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight  * 2] = blue;
			}
		}
		FreeImage_Unload(dib);
	}
}

int main()
{
	OrtEnv* env = NULL;
	CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &env));
	// initialize session options if needed
	OrtSessionOptions* session_options;
	CheckStatus(g_ort->CreateSessionOptions(&session_options));
	// If you have CUDA ONNXRuntime installed otherwise don't use this line
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
	g_ort->SetIntraOpNumThreads(session_options, 8);
	g_ort->SetInterOpNumThreads(session_options, 8);
	OrtSession* session = NULL;
	CheckStatus(g_ort->CreateSession(env, L"facesegmentation_344.onnx", session_options, &session));

	size_t num_input_nodes;
	OrtStatus* status = NULL;
	OrtAllocator* allocator = NULL;
	CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
	CheckStatus(g_ort->SessionGetInputCount(session, &num_input_nodes));
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;
	fprintf(stderr, "Number of inputs = %zu\n", num_input_nodes);
	for (size_t i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name;
		CheckStatus(g_ort->SessionGetInputName(session, i, allocator, &input_name));
		printf("Input %zu : name=%s\n", i, input_name);
		input_node_names[i] = input_name;
		// print input node types
		OrtTypeInfo* typeinfo;
		CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
		const OrtTensorTypeAndShapeInfo* tensor_info;
		g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
		ONNXTensorElementDataType type;
		g_ort->GetTensorElementType(tensor_info, &type);
		fprintf(stdout,"Input %zu : type=%d\n", i, type);
		// print input shapes/dims
		size_t num_dims;
		g_ort->GetDimensionsCount(tensor_info, &num_dims);
		fprintf(stdout,"Input %zu : num_dims=%zu\n", i, num_dims);
		input_node_dims.resize(num_dims);
		g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
		for (size_t j = 0; j < num_dims; j++)
			fprintf(stdout,"Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
		g_ort->ReleaseTypeInfo(typeinfo);

	}
    OrtMemoryInfo* memory_info = NULL;

	CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

	std::vector<float> input_tensor_values;
	FILE* fp = fopen("YOURTESTIMAGE.png", "rb");

	if (fp != NULL)
	{
		ProcessInputImage(fp, input_tensor_values, input_node_dims);
		// create input tensor object from data values
		OrtValue* input_tensor = NULL;
		size_t tensor_size = input_tensor_values.size();

   	    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

		int is_tensor;
		g_ort->IsTensor(input_tensor, &is_tensor);
		if (is_tensor)
		{
			size_t numOut;
			g_ort->SessionGetOutputCount(session, &numOut);
			std::vector<char*> output_names;
			std::vector<int64_t> output_node_dims;
			for (size_t i = 0; i < numOut; i++)
			{
				char* output_name;
				g_ort->SessionGetOutputName(session, i, allocator, &output_name);
				output_names.push_back(output_name);
				printf("OutputName(%d)=%s\n", int(i), output_name);
				OrtTypeInfo* typeinfo;
				CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
				const OrtTensorTypeAndShapeInfo* tensor_info;
				g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
				ONNXTensorElementDataType type;
				g_ort->GetTensorElementType(tensor_info, &type);
				printf("Input %zu : type=%d\n", i, type);
				// print input shapes/dims
				size_t num_dims;
				g_ort->GetDimensionsCount(tensor_info, &num_dims);
				printf("Input %zu : num_dims=%zu\n", i, num_dims);
				output_node_dims.resize(num_dims);
				g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);
				for (size_t j = 0; j < num_dims; j++)
					printf("Output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);

				g_ort->ReleaseTypeInfo(typeinfo);
			}
			OrtValue* output_tensors[] = { NULL };
			g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_names.data(), output_names.size(), output_tensors);
			float *dataOut = NULL;
			g_ort->IsTensor(output_tensors[0], &is_tensor);
			OrtTensorTypeAndShapeInfo* tshapeType;
			g_ort->GetTensorTypeAndShape(output_tensors[0], &tshapeType);
			size_t numCount;
			g_ort->GetTensorShapeElementCount(tshapeType, &numCount);
			CheckStatus(g_ort->GetTensorMutableData(output_tensors[0], (void**)&dataOut));

			FreeImageIO io;
			io.read_proc = myReadProc;
			io.write_proc = myWriteProc;
			io.seek_proc = mySeekProc;
			io.tell_proc = myTellProc;
			FIBITMAP *dibOut = FreeImage_Allocate((int)output_node_dims[3], (int)output_node_dims[2], 24);
#pragma omp parallel for
			for (int y = 0; y < output_node_dims[2]; y++)
			{
				float predVal[3];
				BYTE* bits = FreeImage_GetScanLine(dibOut, int(output_node_dims[2] - 1 - y));

				for (int x = 0; x < output_node_dims[3]; x++)
				{
					predVal[0] = dataOut[x + y * output_node_dims[3]];
					predVal[1] = dataOut[x + y * output_node_dims[3] + output_node_dims[2] * output_node_dims[3]];
					predVal[2] = dataOut[x + y * output_node_dims[3] + output_node_dims[2] * output_node_dims[3] * 2];
					float* softMaxed = Softmax(predVal, 3);
					if (softMaxed[0] > softMaxed[1] && softMaxed[0] > softMaxed[2])
					{
						bits[0] = 0;
						bits[1] = 0;
						bits[2] = 0;
					}
					else if (softMaxed[1] > softMaxed[2] && softMaxed[1] > softMaxed[0])
					{
						bits[0] = 255;
						bits[1] = 255;
						bits[2] = 0;
					}
					else if (softMaxed[2] > softMaxed[0] && softMaxed[2] > softMaxed[1])
					{
						bits[0] = 0;
						bits[1] = 255;
						bits[2] = 255;
					}
					bits += 3;
				}
			}
			FreeImage_Save(FIF_PNG, dibOut, "labelimageoutput.png");
			FreeImage_Unload(dibOut);
			g_ort->ReleaseValue(input_tensor);
			g_ort->ReleaseValue(output_tensors[0]);
		}
	}
	g_ort->ReleaseSession(session);
	g_ort->ReleaseSessionOptions(session_options);
	g_ort->ReleaseEnv(env);
	g_ort->ReleaseMemoryInfo(memory_info);
	return 0;
}