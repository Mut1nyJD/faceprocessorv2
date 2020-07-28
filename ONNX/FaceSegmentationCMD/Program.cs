using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics.Tensors;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Drawing.Drawing2D;

namespace FaceSegmentationCMD
{
    class Program
    {

        
        static private int MaxIndex(float[] rawInput)
        {
            int maxN = 0;
            float maxVal = -1E10F;
            for (var n = 0; n < rawInput.Length; n++)
            {
                if (rawInput[n] > maxVal)
                {
                    maxN = n;
                    maxVal = rawInput[n];
                }
            }
            return maxN;
        }

        static private void Softmax(float[] rawInput)
        {
            float sum = 0.0f;
            float[] tempVal = new float[rawInput.Length];
            for (int n = 0; n < rawInput.Length; n++)
            {
                tempVal[n] = (float)System.Math.Exp(rawInput[n]);
                sum += tempVal[n];
            }
            for (int n = 0; n < rawInput.Length; n++)
                rawInput[n] = tempVal[n] / sum;
        }

        static private Image ImageFromRawBgraArray(byte[] arr, int width, int height, System.Drawing.Imaging.PixelFormat pixelFormat)
        {
            var output = new Bitmap(width, height, pixelFormat);
            var rect = new Rectangle(0, 0, width, height);
            var bmpData = output.LockBits(rect, ImageLockMode.ReadWrite, output.PixelFormat);

            // Row-by-row copy
            var arrRowLength = width * Image.GetPixelFormatSize(output.PixelFormat) / 8;
            var ptr = bmpData.Scan0;
            for (var i = 0; i < height; i++)
            {
                Marshal.Copy(arr, i * arrRowLength, ptr, arrRowLength);
                ptr += bmpData.Stride;
            }

            output.UnlockBits(bmpData);
            return output;
        }

        public static Image ResizeImage(Image image, int new_height, int new_width)
        {
            Bitmap new_image = new Bitmap(new_width, new_height);
            Graphics g = Graphics.FromImage((Image)new_image);
            g.InterpolationMode = InterpolationMode.High;
            g.DrawImage(image, 0, 0, new_width, new_height);
            return new_image;
        }

        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                System.Console.WriteLine("Not enough arguments given use FaceSegmentationCMD inputimage outputimage");
            }
            else {
                byte[] REDLABEL = { 0, 0, 0, 255, 0, 255, 255, 255, 128, 255,0 };
                byte[] GREENLABEL = { 0, 255, 0, 0, 255, 255, 0, 255, 128, 192,128 };
                byte[] BLUELABEL = { 0, 0, 255, 0, 255, 0, 255, 255, 128, 192,128 };
                var options = new SessionOptions();
                options.SetSessionGraphOptimizationLevel(2);
                var path = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
                String onnxfile = path + "\\facesegmentation_full_344.onnx";


                InferenceSession session = null;
                try
                {
                    session = new InferenceSession(onnxfile, options);
                } catch (Exception e)
                {
                    System.Console.WriteLine("Could not load ONNX model, because " + e.ToString());
                    return;
                }
                try
                {

                    var bitmap = new BitmapImage(new Uri(args[0]));
                    var bitmapWidth = bitmap.Width;
                    var bitmapHeight = bitmap.Height;
                    var inputMeta = session.InputMetadata;
                    var container = new List<NamedOnnxValue>();
                    int[] inputDim = new int[4];
                    double scaleFactor = 1.0;
                    foreach (var name in inputMeta.Keys)
                    {
                        var dim = inputMeta[name].Dimensions;
                        for (int n = 0; n < dim.Length; n++)
                            inputDim[n] = dim[n];
                    }
                    if (bitmapWidth > bitmapHeight)
                    {
                        scaleFactor = (double)inputDim[3] / bitmapWidth;
                    }
                    else
                    {
                        scaleFactor = (double)inputDim[2] / bitmapHeight;
                    }
                    TransformedBitmap tb = new TransformedBitmap(bitmap, new System.Windows.Media.ScaleTransform(scaleFactor, scaleFactor));
                    int newWidth = tb.PixelWidth;
                    int newHeight = tb.PixelHeight;
                    int channels = tb.Format.BitsPerPixel / 8;
                    int stride = channels * newWidth;
                    byte[] rawData = new byte[stride * newHeight];
                    byte[] rawLabelOutput = new byte[inputDim[2] * inputDim[3]];
                    byte[] rawOutput = new byte[stride * newHeight];
                    tb.CopyPixels(rawData, stride, 0);
                    int paddingX = inputDim[3] - newWidth;
                    int paddingY = inputDim[2] - newHeight;
                    float[] testData = new float[inputDim[2] * inputDim[3] * inputDim[1]];
                    // intialize the whole tensor data to background value so do not have to deal with padding later
                    for (int n = 0; n < inputDim[2] * inputDim[3] * inputDim[1]; n++)
                        testData[n] = -1.0f;
                    var offsetX = paddingX / 2;
                    var offsetY = paddingY / 2;
                    // fill up tensor with image data                    
                    for (int y = 0; y < newHeight; y++)
                    {
                        int y1 = y;
                        for (int x = 0; x < newWidth; x++)
                        {
                            testData[(x + offsetX) + (y + offsetY) * inputDim[3] + inputDim[2] * inputDim[3] * 2] = rawData[(x + y1 * newWidth) * channels] / 127.5f;
                            testData[(x + offsetX) + (y + offsetY) * inputDim[3] + inputDim[2] * inputDim[3]] = rawData[(x + y1 * newWidth) * channels + 1] / 127.5f;
                            testData[(x + offsetX) + (y + offsetY) * inputDim[3]] = rawData[(x + y1 * newWidth) * channels + 2] / 127.5f;
                            testData[(x + offsetX) + (y + offsetY) * inputDim[3] + inputDim[2] * inputDim[3] * 2] -= 1.0f;
                            testData[(x + offsetX) + (y + offsetY) * inputDim[3] + inputDim[2] * inputDim[3]] -= 1.0f;
                            testData[(x + offsetX) + (y + offsetY) * inputDim[3]] -= 1.0f;
                        }
                    }
                    foreach (var name in inputMeta.Keys)
                    {
                        var tensor = new DenseTensor<float>(testData, inputMeta[name].Dimensions);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                    using (var results = session.Run(container))
                    {
                        int numResults = results.Count;
                        foreach (var r in results)
                        {
                            System.Console.WriteLine(r.Name);
                            var resultTensor = r.AsTensor<float>();
                            var resultDimension = resultTensor.Dimensions;
                            System.Console.WriteLine(resultDimension.Length);
                            var resultArray = resultTensor.ToArray();
                            float[] pointVal = new float[resultDimension[1]];
                            for (var y = 0; y < resultDimension[2]; y++)
                            {
                                for (var x = 0; x < resultDimension[3]; x++)
                                {
                                    for (var n = 0; n < resultDimension[1]; n++)
                                    {
                                        pointVal[n] = resultArray[x + y * resultDimension[3] + n * resultDimension[2] * resultDimension[3]];
                                    }
                                    Softmax(pointVal);
                                    byte labelVal = (byte)MaxIndex(pointVal);
                                    rawLabelOutput[x + y * resultDimension[3]] = labelVal;
                                }
                            }
                            if (resultDimension[1] < 64)
                            {
                                for (int y = 0; y < newHeight; y++)
                                {
                                    for (int x = 0; x < newWidth; x++)
                                    {
                                        int n = rawLabelOutput[(x + offsetX) + (y + offsetY) * resultDimension[3]];
                                        rawOutput[(x + y * newWidth) * channels + 3] = 255;
                                        rawOutput[(x + y * newWidth) * channels + 2] = REDLABEL[n];
                                        rawOutput[(x + y * newWidth) * channels + 1] = GREENLABEL[n];
                                        rawOutput[(x + y * newWidth) * channels] = BLUELABEL[n];
                                    }
                                }
                            } else
                            {
                                for (int y = 0; y < newHeight; y++)
                                {
                                    for (int x = 0; x < newWidth; x++)
                                    {
                                        int n = rawLabelOutput[(x + offsetX) + (y + offsetY) * resultDimension[3]];
                                        rawOutput[(x + y * newWidth) * channels + 3] = 255;
                                        rawOutput[(x + y * newWidth) * channels + 2] = (byte)n;
                                        rawOutput[(x + y * newWidth) * channels + 1] = (byte)n;
                                        rawOutput[(x + y * newWidth) * channels] = (byte)n;
                                    }
                                }
                            }
                            var outputImage = ImageFromRawBgraArray(rawOutput, newWidth, newHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                            outputImage = ResizeImage(outputImage, (int)bitmapHeight, (int)bitmapWidth);
                            outputImage.Save(args[1]);
                        }
                    }
                }
                catch (Exception e)
                {
                    System.Console.WriteLine("Could not load image because of " + e.ToString());
                }
            }
        }
    }
}
