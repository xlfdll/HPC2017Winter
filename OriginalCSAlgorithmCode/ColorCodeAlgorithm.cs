using System;
using System.Threading;
using System.Windows;

namespace CSS584A1
    {
    public static class ColorCodeAlgorithm
        {
        public static void ProcessPixels(ImageInfo info, Int32Rect area)
            {
            Byte[] pixelData = new Byte[4];

            for (Int32 x = area.X; x < area.Width; x++)
                {
                for (Int32 y = area.Y; y < area.Height; y++)
                    {
                    info.Image.CopyPixels(new Int32Rect(x, y, 1, 1), pixelData, info.Stride, 0);

                    Int32 colorCode =
                        ((pixelData[2] & 0xC0) >> 2)
                        | ((pixelData[1] & 0xC0) >> 4)
                        | ((pixelData[0] & 0xC0) >> 6);
                    
                    // Thread-safe increment
                    Interlocked.Increment(ref info.Features.ColorCodeBins[colorCode]);
                    }
                }
            }

        public static Double GetManhattanDistance(ImageInfo referenceImageInfo, ImageInfo targetImageInfo)
            {
            Double distance = 0.0;

            for (Int32 i = 0; i < ColorCodeAlgorithm.BinCount; i++)
                {
                distance += Math.Abs(Convert.ToDouble(referenceImageInfo.Features.ColorCodeBins[i]) / (referenceImageInfo.Image.PixelWidth * referenceImageInfo.Image.PixelHeight)
                    - Convert.ToDouble(targetImageInfo.Features.ColorCodeBins[i]) / (targetImageInfo.Image.PixelWidth * targetImageInfo.Image.PixelHeight));
                }

            return distance;
            }

        public const Int32 BinCount = 64;
        }
    }