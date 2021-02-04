// model_execution.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <omp.h>
#include <iostream>
#include<vector>
#include<list>
#include<omp.h>
// FREEImage library (https://freeimage.sourceforge.io/index.html)
#include "Freeimage.h"
// ONNXRuntime (https://github.com/Microsoft/onnxruntime)
#include "onnxruntime_c_api.h"



class DetRect
{
	public:
		float m_x1;
		float m_y1;
		float m_x2;
		float m_y2;
		float m_score;
        unsigned char m_classNr;

		bool operator <(DetRect& other)
		{
			return (other.m_score < this->m_score);
		}

		bool operator >(DetRect& other)
		{
			return (other.m_score > this->m_score);
		}

		DetRect Intersect(DetRect& other)
		{
			DetRect inter;
			inter.m_x1 = this->m_x1 > other.m_x1 ? this->m_x1 : other.m_x1;
			inter.m_y1 = this->m_y1 > other.m_y1 ? this->m_y1 : other.m_y1;
			inter.m_x2 = this->m_x2 < other.m_x2 ? this->m_x2 : other.m_x2;
			inter.m_y2 = this->m_y2 < other.m_y2 ? this->m_y2 : other.m_y2;
			return inter;
		}

		float Area() const
		{
			if (m_x1 > m_x2 || m_y1 > m_y2)
				return 0.0f;
			return (this->m_x2 - this->m_x1) * (this->m_y2 - this->m_y1);
		}
};


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
const float g_scales[] = { 32.0f,16.0f,8.0f };
const float SCORETHRESHOLD = 0.4f;
const float IoUTHRESHOLD = 0.5f;
std::pair<float, float> g_anchorLevels[3][3];

void CheckStatus(OrtStatus* status)
{
	if (status != NULL) {
		const char* msg = g_ort->GetErrorMessage(status);
		fprintf(stderr, msg);
	}
}

inline float Sigmoid(float input)
{
	return 1.0f / (1 + expf(-input));
}

inline float SQR(float input)
{
	return input * input;
}

void DrawRect(DetRect& rect, unsigned char* buffer, unsigned int imgWidth, unsigned int imgHeight, unsigned int nChannels, unsigned int lineWidth)
{
	int startX = (int)rect.m_x1;
	if (startX < 0)
		startX = 0;
	int endX = (int)rect.m_x2;
	if (endX > (int)imgWidth - 1)
		endX = (int)imgWidth - 1;
	int startY = (int)rect.m_y1;
	if (startY < 0)
		startY = 0;
	int endY = (int)rect.m_y2;
	if (endY > (int)imgHeight - 1)
		endY = (int)imgHeight - 1;

	int imgStride = int(imgWidth * nChannels);

#pragma omp parallel for
	for (int line = 0; line < lineWidth; line++)
	{
		int yPos = startY + line;
		if (yPos >= 0 && yPos < imgHeight - 1)
		{
			for (int x = startX; x <= endX; x++)
			{
				buffer[x*nChannels + yPos * imgStride] = 0;
				buffer[x*nChannels + yPos * imgStride + 1] = 255;
				buffer[x*nChannels + yPos * imgStride + 2] = 0;
			}
		}
		yPos = endY + line;
		if (yPos >= 0 && yPos < imgHeight - 1)
		{
			for (int x = startX; x <= endX; x++)
			{
				buffer[x*nChannels + yPos * imgStride] = 0;
				buffer[x*nChannels + yPos * imgStride + 1] = 255;
				buffer[x*nChannels + yPos * imgStride + 2] = 0;
			}
		}
		int xPos = startX + line;
		if (xPos >= 0 && xPos < imgWidth - 1)
		{
			for (int y = startY; y <= endY; y++)
			{
				buffer[xPos * nChannels + y * imgStride] = 0;
				buffer[xPos * nChannels + y * imgStride + 1] = 255;
				buffer[xPos * nChannels + y * imgStride + 2] = 0;
			}
		}
		xPos = endX + line;
		if (xPos >= 0 && xPos < imgWidth - 1)
		{
			for (int y = startY; y <= endY; y++)
			{
				buffer[xPos * nChannels + y * imgStride] = 0;
				buffer[xPos * nChannels + y * imgStride + 1] = 255;
				buffer[xPos * nChannels + y * imgStride + 2] = 0;
			}
		}
	}
}



void DrawOutputImage(FILE* fp, std::vector<DetRect>& detRects,std::string& outfilename)
{
	FreeImageIO io;
	io.read_proc = myReadProc;
	io.write_proc = myWriteProc;
	io.seek_proc = mySeekProc;
	io.tell_proc = myTellProc;
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromHandle(&io, (fi_handle)fp, 0);
	if (fif != FIF_UNKNOWN)
	{
		FIBITMAP *dib = FreeImage_LoadFromHandle(fif, &io, (fi_handle)fp, 0);
		unsigned int height = FreeImage_GetHeight(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		unsigned char* outBuffer = new unsigned char[width*height*(bpp == 32 ? 4 : 3)];
		unsigned int nChannels = (bpp == 32 ? 4 : 3);
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			// Y-Direction is flipped
			int srcY = height - 1 - y;
			BYTE* bits = FreeImage_GetScanLine(dib, srcY);

			for (int x = 0; x < width; x++)
			{
				for (int n = 0; n < nChannels; n++)
				{
					outBuffer[(x + y * width) * nChannels + n] = bits[n];
				}
				bits += nChannels;
			}
		}
		FreeImage_Unload(dib);
		std::vector<DetRect>::const_iterator iter;
		for (iter = detRects.begin(); iter != detRects.end(); iter++)
		{
			DetRect rect = *iter;
			DrawRect(rect, outBuffer, width, height, nChannels, 3);
		}
		FIBITMAP* dibOut = FreeImage_Allocate(width, height, bpp);
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			// Y-Direction is flipped
			int srcY = height - 1 - y;
			BYTE* bits = FreeImage_GetScanLine(dibOut, srcY);

			for (int x = 0; x < width; x++)
			{
				for (int n = 0; n < nChannels; n++)
				{
					bits[n] = outBuffer[(x + y * width) * nChannels + n];
				}
				bits += nChannels;
			}
		}
		FreeImage_Save(FIF_PNG, dibOut, outfilename.c_str());
	}
}

void ProcessInputImage(FILE* fp, std::vector<float>& output, std::vector<int64_t>& inDims, float& scale, __int64& deltaX, __int64& deltaY)
{
	FreeImageIO io;
	io.read_proc = myReadProc;
	io.write_proc = myWriteProc;
	io.seek_proc = mySeekProc;
	io.tell_proc = myTellProc;
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromHandle(&io, (fi_handle)fp, 0);
	if (fif != FIF_UNKNOWN)
	{
		FIBITMAP *dib = FreeImage_LoadFromHandle(fif, &io, (fi_handle)fp, 0);
		unsigned int height = FreeImage_GetHeight(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
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

		scale = float(width) / float(newWidth);
		__int64 outWidth = inDims[3];
		__int64 outHeight = inDims[2];
		output.resize(outWidth*outHeight * 3);
		for (int n = 0; n < outWidth * outHeight * 3; n++)
			output[n] = 0.0f;
		deltaX = (outWidth - newWidth) >> 1;
		deltaY = (outHeight - newHeight) >> 1;

#pragma omp parallel for
		for (int y = 0; y < newHeight; y++)
		{
			// Y-Direction is flipped
			int srcY = height - 1 - (int)(y * scale);
			BYTE* bits = FreeImage_GetScanLine(dib, srcY);

			for (int x = 0; x < newWidth; x++)
			{
				int srcX = int(x*scale);
				float red = float(bits[srcX * (bpp == 32 ? 4 : 3) + 2]);
				red /= 255.0f;
				float green = float(bits[srcX * (bpp == 32 ? 4 : 3) + 1]);
				green /= 255.0f;
				float blue = float(bits[srcX * (bpp == 32 ? 4 : 3)]);
				blue /= 255.0f;
				output[((x + deltaX) + (y + deltaY) * outWidth)] = red;
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight] = green;
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight * 2] = blue;
			}
		}
		FreeImage_Unload(dib);
	}
}


void ProcessOutput(int levelNr, float* tensorData, std::vector<int64_t>& resultDims, std::list<DetRect>& detRects)
{
	int boxStride = (int)(resultDims[2] * resultDims[3] * resultDims[4]);
    int numClasses = resultDim[4] - 5;
	for (int boxNr = 0; boxNr < resultDims[1]; boxNr++)
	{
		for (int gridY = 0; gridY < resultDims[2]; gridY++)
		{
			for (int gridX = 0; gridX < resultDims[3]; gridX++)
			{
				float bx = (Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4]]) * 2 - 0.5f + gridX) * g_scales[levelNr];
				float by = (Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 1]) * 2 -0.5f + gridY) * g_scales[levelNr];
				float bw = SQR(Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 2]) * 2) * g_anchorLevels[levelNr][boxNr].first;
				float bh = SQR(Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 3]) * 2) * g_anchorLevels[levelNr][boxNr].second;
				float obj = Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 4]);
                float maxClsScr = 0.0f;
                unsigned char classNr = 0;
                for (int cls = 0; cls < numClasses; cls++)
                {
				    float classScr = Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 5+cls]);
				    if (classScr > maxClsScr)
				    {
				       maxClsScr = classScr;
				       classNr = cls;
				    }
				}
				if (obj * maxClassScr > SCORETHRESHOLD)
				{
					DetRect det;
					det.m_x1 = bx - 0.5f * bw;
					det.m_x2 = bx + 0.5f * bw;
					det.m_y1 = by - 0.5f * bh;
					det.m_y2 = by + 0.5f * bh;
					det.m_score = obj * classScr;
					det.m_classNr = classNr;
					detRects.push_back(det);
				}
			}
		}
	}
}


std::vector<DetRect> DoNMS(std::list<DetRect>& dets, float scaleFactor, __int64 offsetX, __int64 offsetY)
{
	dets.sort();
	std::vector<DetRect> finalRects;
	std::list<DetRect>::const_iterator iter;
	int n = 0;
	for (iter = dets.begin(); iter != dets.end(); iter++, n++)
	{
		bool survived = true;
		DetRect cur = *iter;
		std::list<DetRect>::const_iterator iter2;
		int n1 = 0;
		for (iter2 = dets.begin(); iter2 != dets.end(); iter2++, n1++) if (n1 > n)
		{
			DetRect other = *iter2;
			DetRect intersect = other.Intersect(cur);
			float interArea = intersect.Area();
			float IoU = interArea / other.Area();
			if (IoU > IoUTHRESHOLD)
			{
				survived = false;
				break;
			}
		}
		if (survived)
		{
			finalRects.push_back(cur);
		}
	}
	// project back into original image size
	for (int n = 0; n < finalRects.size(); n++)
	{
		DetRect& rect = finalRects[n];
		rect.m_x1 = (rect.m_x1 - offsetX) * (float)scaleFactor;
		rect.m_x2 = (rect.m_x2 - offsetX) * (float)scaleFactor;
		rect.m_y1 = (rect.m_y1 - offsetY) * (float)scaleFactor;
		rect.m_y2 = (rect.m_y2 - offsetY) * (float)scaleFactor;
	}
	return finalRects;
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "Not enough arguments provided use input_image output_image\n");
		return -1;
	}
	std::string outfilename = std::string(argv[2]);
	OrtEnv* env = NULL;
	CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &env));
	// initialize session options if needed
	OrtSessionOptions* session_options;
	CheckStatus(g_ort->CreateSessionOptions(&session_options));
	g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
	g_ort->SetIntraOpNumThreads(session_options, 8);
	g_ort->SetInterOpNumThreads(session_options, 8);
	OrtSession* session = NULL;
	CheckStatus(g_ort->CreateSession(env, L"PLACE_MUT1NYDETMODELHERE.onnx", session_options, &session));
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
		fprintf(stdout, "Input %zu : type=%d\n", i, type);
		// print input shapes/dims
		size_t num_dims;
		g_ort->GetDimensionsCount(tensor_info, &num_dims);
		fprintf(stdout, "Input %zu : num_dims=%zu\n", i, num_dims);
		input_node_dims.resize(num_dims);
		g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
		for (size_t j = 0; j < num_dims; j++)
			fprintf(stdout, "Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
		g_ort->ReleaseTypeInfo(typeinfo);

	}
	OrtMemoryInfo* memory_info = NULL;
	CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	std::vector<float> input_tensor_values;
	FILE* fp = fopen(argv[1], "rb");
	if (fp != NULL)
	{
		float scale;
		__int64 xOffset, yOffset;
		ProcessInputImage(fp, input_tensor_values, input_node_dims,scale,xOffset,yOffset);
		fclose(fp);
		// create input tensor object from data values
		OrtValue* input_tensor = NULL;
		size_t tensor_size = input_tensor_values.size();
		CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
		int is_tensor;
		g_ort->IsTensor(input_tensor, &is_tensor);
		if (is_tensor)
		{

			g_anchorLevels[0][0] = std::pair<float, float>(116.0f, 90.0f);
			g_anchorLevels[0][1] = std::pair<float, float>(156.0f, 198.0f);
			g_anchorLevels[0][2] = std::pair<float,float>(373.0f, 326.0f);
			g_anchorLevels[1][0] = std::pair<float,float>(30.0f,61.0f);
			g_anchorLevels[1][1] = std::pair<float, float>(63.0f, 45.0f);
			g_anchorLevels[1][2] = std::pair<float, float>(59.0f, 119.0f);
			g_anchorLevels[2][0] = std::pair<float, float>(10.0f, 13.0f);
			g_anchorLevels[2][1] = std::pair<float, float>(16.0f, 30.0f);
			g_anchorLevels[2][2] = std::pair<float, float>(33.0f, 23.0f);

			size_t numOut;
			g_ort->SessionGetOutputCount(session, &numOut);
			std::vector<char*> output_names;
			std::vector<std::vector<int64_t> > output_nodes_dims;
			std::vector<int64_t>  output_node_dims;
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
				output_nodes_dims.push_back(output_node_dims);
			}
			OrtValue* output_tensors[] = { NULL,NULL,NULL };
			g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_names.data(), output_names.size(), output_tensors);
			float *dataOut = NULL;
			std::list<DetRect> detRects;
			for (int level = 0; level < 3; level++)
			{
				g_ort->IsTensor(output_tensors[level], &is_tensor);
				if (is_tensor)
				{
					CheckStatus(g_ort->GetTensorMutableData(output_tensors[level], (void**)&dataOut));
					ProcessOutput(level, dataOut, output_nodes_dims[level], detRects);
				}
			}
			fprintf(stderr, "# dets = %d \n", (int)detRects.size());
			std::vector<DetRect> finalRects = DoNMS(detRects, scale, xOffset, yOffset);
			fprintf(stderr, "# dets = %d \n", (int)finalRects.size());
			fp = fopen(argv[1], "rb");
			DrawOutputImage(fp, finalRects, outfilename);
			fclose(fp);
		}
	}

	return 0;
}

