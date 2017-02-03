using System;
using System.Threading;
using System.Windows;

namespace CSS584A1
    {
    public static class IntensityAlgorithm
        {
        public static void ProcessPixels(ImageInfo info, Int32Rect area)
            {
            Byte[] pixelData = new Byte[4];

            for (Int32 x = area.X; x < area.Width; x++)
                {
                for (Int32 y = area.Y; y < area.Height; y++)
                    {
                    info.Image.CopyPixels(new Int32Rect(x, y, 1, 1), pixelData, info.Stride, 0);

                    // Thread-safe increment
                    Interlocked.Increment(ref info.Features.IntensityBins[IntensityAlgorithm.GetHistogramBinIndex(pixelData[2], pixelData[1], pixelData[0])]);
                    }
                }
            }

        public static Double GetManhattanDistance(ImageInfo referenceImageInfo, ImageInfo targetImageInfo)
            {
            Double distance = 0.0;

            for (Int32 i = 0; i < IntensityAlgorithm.BinCount; i++)
                {
                distance += Math.Abs(Convert.ToDouble(referenceImageInfo.Features.IntensityBins[i]) / (referenceImageInfo.Image.PixelWidth * referenceImageInfo.Image.PixelHeight)
                    - Convert.ToDouble(targetImageInfo.Features.IntensityBins[i]) / (targetImageInfo.Image.PixelWidth * targetImageInfo.Image.PixelHeight));
                }

            return distance;
            }

        private static Int32 GetHistogramBinIndex(Byte r, Byte g, Byte b)
            {
            Int32 i = (Int32)(0.299 * r + 0.587 * g + 0.114 * b);
            Int32 index = i / 10;

            if (index > IntensityAlgorithm.BinCount - 1)
                {
                index = IntensityAlgorithm.BinCount - 1;
                }

            return index;
            }

        public const Int32 BinCount = 25;
        }
    }