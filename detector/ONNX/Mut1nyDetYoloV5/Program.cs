using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Windows.Media.Imaging;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections;
using System.IO;
using System.Diagnostics;

namespace DetYoloV5
{

    
    public class RectCompare : IComparer
    {
        int IComparer.Compare(object a, object b)
        {
            if (a is DetRect && b is DetRect)
            {
                return (a as DetRect).m_score > (b as DetRect).m_score ? 1 : -1;
            }
            throw new ArgumentException("Objects are not DetRects");
        }

    }

    public class DetRect : IComparable
    {
        public float m_x1;
        public float m_y1;
        public float m_x2;
        public float m_y2;
        public float m_score;
        
        public int CompareTo(object obj)
        {
            if (obj is DetRect)
            {
                return this.m_score > (obj as DetRect).m_score ? 1 : -1;
            }
            throw new ArgumentException("Object is not a DetRect");
        }

        public DetRect Intersect(DetRect other)
        {
            DetRect inter = new DetRect();
            inter.m_x1 = this.m_x1 > other.m_x1 ? this.m_x1 : other.m_x1;
            inter.m_y1 = this.m_y1 > other.m_y1 ? this.m_y1 : other.m_y1;
            inter.m_x2 = this.m_x2 < other.m_x2 ? this.m_x2 : other.m_x2;
            inter.m_y2 = this.m_y2 < other.m_y2 ? this.m_y2 : other.m_y2;
            return inter;
        }

        public float Area()
        {
            if (m_x1 > m_x2 || m_y1 > m_y2)
                return 0.0f;
            return (this.m_x2 - this.m_x1) * (this.m_y2 - this.m_y1);
        }

        public bool IsInside(DetRect other)
        {
            return other.m_x1 >= this.m_x1 && other.m_x2 <= this.m_x2 && other.m_y1 >= this.m_y1 && other.m_y2 <= this.m_y2; 
        }
    }


    class Program
    {

        static private float[,,] anchorLevels = new float[3, 3, 2];
        static private float[] scales = new float[3];
        static private float IoUTHRESHOLD = 0.5f;
        static private float SCORETHRESHOLD = 0.4f;
        static private bool BENCHMARKMODE = false;
        static private int NUMRUNS = 500;
        static private int WARMUPRUNS = 10;

        static private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)(Math.Exp(-x)));
        }

        static private float SQR(float x)
        {
            return x * x;
        }

        static float[] CreateInputTensorFromImage(String filename, int[] tensorDims, out float scaleFactor, out int offsetX, out int offsetY)
        {
            var bitmap = new BitmapImage(new Uri(filename));
            var bitmapWidth = bitmap.PixelWidth;
            var bitmapHeight = bitmap.PixelHeight;
            scaleFactor = 1.0f;
            if (bitmapWidth > bitmapHeight)
            {
                scaleFactor = (float)tensorDims[3] / bitmapWidth;
            }
            else
            {
                scaleFactor = (float)tensorDims[2] / bitmapHeight;
            }
            TransformedBitmap tb = new TransformedBitmap(bitmap, new System.Windows.Media.ScaleTransform(scaleFactor, scaleFactor));
            int newWidth = tb.PixelWidth;
            int newHeight = tb.PixelHeight;
            int channels = tb.Format.BitsPerPixel / 8;
            int stride = channels * newWidth;
            byte[] rawData = new byte[stride * newHeight];
            byte[] rawLabelOutput = new byte[tensorDims[2] * tensorDims[3]];
            byte[] rawOutput = new byte[stride * newHeight];
            tb.CopyPixels(rawData, stride, 0);
            int paddingX = tensorDims[3] - newWidth;
            int paddingY = tensorDims[2] - newHeight;
            float[] testData = new float[tensorDims[2] * tensorDims[3] * tensorDims[1]];
            for (int n = 0; n < tensorDims[2] * tensorDims[3] * tensorDims[1]; n++)
                testData[n] = 0.0f;
            offsetX = paddingX / 2;
            offsetY = paddingY / 2;
            // fill up tensor with image data                    
            for (int y = 0; y < newHeight; y++)
            {
                int y1 = y;
                for (int x = 0; x < newWidth; x++)
                {
                    testData[(x + offsetX) + (y + offsetY) * tensorDims[3] + tensorDims[2] * tensorDims[3] * 2] = rawData[(x + y1 * newWidth) * channels] / 255.0f;
                    testData[(x + offsetX) + (y + offsetY) * tensorDims[3] + tensorDims[2] * tensorDims[3]] = rawData[(x + y1 * newWidth) * channels + 1] / 255.0f;
                    testData[(x + offsetX) + (y + offsetY) * tensorDims[3]] = rawData[(x + y1 * newWidth) * channels + 2] / 255.0f;
                }
            }
            return testData;
        }

        static void ProcessOutput(int levelNr, ReadOnlySpan<int> resultDims, float[] tensorData, ArrayList dets)
        {
            int boxStride = resultDims[2] * resultDims[3] * resultDims[4];

            for (int boxNr = 0; boxNr < resultDims[1]; boxNr++)
            {
                for (int gridY = 0; gridY < resultDims[2]; gridY++)
                {
                    for (int gridX = 0; gridX < resultDims[3]; gridX++)
                    {
                        var bx = (Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4]]) * 2 - 0.5f + gridX) * scales[levelNr];
                        var by = (Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 1]) * 2 - 0.5f + gridY) * scales[levelNr];
                        var bw = (float)SQR(Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 2]) * 2) * anchorLevels[levelNr, boxNr, 0];
                        var bh = (float)SQR(Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 3]) * 2) * anchorLevels[levelNr, boxNr, 1];
                        var obj = Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 4]);
                        var classScr = Sigmoid(tensorData[boxNr * boxStride + (gridX + gridY * resultDims[3]) * resultDims[4] + 5]);
                        if (obj * classScr > SCORETHRESHOLD)
                        {
                            DetRect det = new DetRect();
                            det.m_x1 = bx - 0.5f * bw;
                            det.m_x2 = bx + 0.5f * bw;
                            det.m_y1 = by - 0.5f * bh;
                            det.m_y2 = by + 0.5f * bh;
                            det.m_score = obj * classScr;
                            dets.Add(det);
                        }
                    }
                }
            }
        }

        static ArrayList BuildFinalOutput(ArrayList dets, float scaleFactor, int offsetX, int offsetY)
        {
            ArrayList filteredRects = new ArrayList();
            for (int n = 0; n < dets.Count; n++)
            {
                bool survived = true;
                DetRect cur = (DetRect)dets[n];
                for (int n1 = n + 1; n1 < dets.Count; n1++)
                {
                    DetRect other = (DetRect)dets[n1];
                    DetRect intersect = other.Intersect(cur);
                    var interArea = intersect.Area();
                    var IoU = interArea / other.Area();
                    if (IoU > IoUTHRESHOLD)
                    {
                        survived = false;
                        break;
                    }
                }
                if (survived)
                {
                    filteredRects.Add(cur);
                }
            }
            // project back into original image size
            for (int n = 0; n < filteredRects.Count; n++)
            {
                DetRect rect = (DetRect)filteredRects[n];
                rect.m_x1 = (rect.m_x1 - offsetX) * (float)scaleFactor;
                rect.m_x2 = (rect.m_x2 - offsetX) * (float)scaleFactor;
                rect.m_y1 = (rect.m_y1 - offsetY) * (float)scaleFactor;
                rect.m_y2 = (rect.m_y2 - offsetY) * (float)scaleFactor;
            }
            return filteredRects;
        }

        static void DrawRectangle(DetRect rect, byte[] buffer, int lineWidth, int imgWidth, int imgHeight, int imgStride, int nChannels)
        {
            int startX = (int)rect.m_x1;
            if (startX < 0)
                startX = 0;
            int endX = (int)rect.m_x2;
            if (endX > imgWidth - 1)
                endX = imgWidth - 1;
            int startY = (int)rect.m_y1;
            if (startY < 0)
                startY = 0;
            int endY = (int)rect.m_y2;
            if (endY > imgHeight - 1)
                endY = imgHeight - 1;

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
                if (xPos >=0 && xPos < imgWidth-1)
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

        static void WriteOutputDet(String inputFilename, String outputFilename, ArrayList detRects)
        {
            var bitmap = new BitmapImage(new Uri(inputFilename));
            var bitmapWidth = bitmap.PixelWidth;
            var bitmapHeight = bitmap.PixelHeight;
            int channels = bitmap.Format.BitsPerPixel / 8;
            int stride = channels * bitmapWidth;
            byte[] rawData = new byte[stride * bitmapHeight];
            bitmap.CopyPixels(rawData, stride, 0);
           foreach (var detRect in detRects)
            {
                DrawRectangle((DetRect)detRect, rawData,3,bitmapWidth,bitmapHeight,stride, channels);
            }
            var outbitmap = BitmapSource.Create(bitmapWidth, bitmapHeight, 96, 96, bitmap.Format, null, rawData, stride);
            using (var fileStream = new FileStream(outputFilename, FileMode.Create))
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(outbitmap));
                encoder.Save(fileStream);
            }
        }

        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                System.Console.WriteLine("Not enough arguments given, use input image output image");
                return;
            }
            // when using CPU / MKLDNN provider uncomment the next line
            // var options = new SessionOptions();
            // when using CUDA/GPU Provider if not comment this line
            var options = SessionOptions.MakeSessionOptionWithCudaProvider();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.InterOpNumThreads = 8;
            options.IntraOpNumThreads = 8;
            String onnxfile = "YOUR_MUT1NY_DETECTOR_MODEL.onnx";
            InferenceSession session = null;
            scales[0] = 32.0f;
            scales[1] = 16.0f;
            scales[2] = 8.0f;
            anchorLevels[0, 0, 0] = 116.0f;
            anchorLevels[0, 0, 1] = 90.0f;
            anchorLevels[0, 1, 0] = 156.0f;
            anchorLevels[0, 1, 1] = 198.0f;
            anchorLevels[0, 2, 0] = 373.0f;
            anchorLevels[0, 2, 1] = 326.0f;

            anchorLevels[1, 0, 0] = 30.0f;
            anchorLevels[1, 0, 1] = 61.0f;
            anchorLevels[1, 1, 0] = 62.0f;
            anchorLevels[1, 1, 1] = 45.0f;
            anchorLevels[1, 2, 0] = 59.0f;
            anchorLevels[1, 2, 1] = 119.0f;

            anchorLevels[2, 0, 0] = 10.0f;
            anchorLevels[2, 0, 1] = 13.0f;
            anchorLevels[2, 1, 0] = 16.0f;
            anchorLevels[2, 1, 1] = 30.0f;
            anchorLevels[2, 2, 0] = 33.0f;
            anchorLevels[2, 2, 1] = 23.0f;
            String inputFilename = args[0];
            String outputFilename = args[1];

            try
            {
                session = new InferenceSession(onnxfile, options);
                var inputMeta = session.InputMetadata;
                int[] inputDim = new int[4];
                float scaleFactor;
                int offsetX, offsetY;
                foreach (var name in inputMeta.Keys)
                {
                    var dim = inputMeta[name].Dimensions;
                    for (int n = 0; n < dim.Length; n++)
                        inputDim[n] = dim[n];
                }
                Stopwatch totalSw;
                Stopwatch processInSw;
                Stopwatch processOutSw;
                Stopwatch executeSw;
                long totalTime = 0;
                long totalProcessIn = 0;
                long totalProcessOut = 0;
                long totalExecute = 0;
                for (int runs = 0; runs < (BENCHMARKMODE ? NUMRUNS : 1); runs++)
                { 
                    totalSw = Stopwatch.StartNew();
                    processInSw = Stopwatch.StartNew();
                    var testData = CreateInputTensorFromImage(inputFilename, inputDim, out scaleFactor, out offsetX, out offsetY);
                    processInSw.Stop();
                    var container = new List<NamedOnnxValue>();

                    foreach (var name in inputMeta.Keys)
                    {
                        var tensor = new DenseTensor<float>(testData, inputMeta[name].Dimensions);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                    executeSw = Stopwatch.StartNew();
                    using (var results = session.Run(container))
                    {
                        executeSw.Stop();
                        int numResults = results.Count;
                        int levelNr = 0;
                        ArrayList dets = new ArrayList();
                        processOutSw = Stopwatch.StartNew();
                        foreach (var r in results)
                        {
                            var resultTensor = r.AsTensor<float>();
                            var resultDimension = resultTensor.Dimensions;
                            var resultArray = resultTensor.ToArray();
                            ProcessOutput(levelNr, resultDimension, resultArray, dets);
                            levelNr++;
                        }
                        System.Console.WriteLine("# Dets = " + dets.Count);
                        processOutSw.Stop();
                        dets.Sort();
                        ArrayList finalRects = BuildFinalOutput(dets, 1.0f / scaleFactor, offsetX, offsetY);
                        System.Console.WriteLine("Final # detected Rects = " + finalRects.Count);
                        totalSw.Stop();
                        Console.WriteLine("Prepocessing took " + processInSw.ElapsedMilliseconds);
                        Console.WriteLine("Execution of DNN took " + executeSw.ElapsedMilliseconds);
                        Console.WriteLine("Postprocessing took " + processOutSw.ElapsedMilliseconds);
                        Console.WriteLine("Total processing took " + totalSw.ElapsedMilliseconds);
                        if (runs > WARMUPRUNS)
                        {
                            totalTime += totalSw.ElapsedMilliseconds;
                            totalExecute += executeSw.ElapsedMilliseconds;
                            totalProcessIn += processInSw.ElapsedMilliseconds;
                            totalProcessOut += processOutSw.ElapsedMilliseconds;
                        }
                        if (!BENCHMARKMODE)
                            WriteOutputDet(inputFilename, outputFilename, finalRects);
                        results.Dispose();
                        container.Clear();
                       
                    }
                }
                float avgTotalTime = (float)totalTime / (float)(NUMRUNS - WARMUPRUNS);
                float avgExecuteTime = (float)totalExecute / (float)(NUMRUNS - WARMUPRUNS);
                float avgProcessInTime = (float)totalProcessIn / (float)(NUMRUNS - WARMUPRUNS);
                float avgProcessOutTime = (float)totalProcessOut / (float)(NUMRUNS - WARMUPRUNS);
                Console.WriteLine("Avg time of preprocess: " + avgProcessInTime);
                Console.WriteLine("Avg time of xecution of DNN: " + avgExecuteTime);
                Console.WriteLine("Avg time of postprocess: " + avgProcessOutTime);
                Console.WriteLine("Avg time of total: " + avgTotalTime);

            }
            catch (Exception e)
            {
                System.Console.WriteLine("Could not load ONNX model, because " + e.ToString());
                return;
            }
            System.Console.WriteLine("Done");
        }
    }
}
