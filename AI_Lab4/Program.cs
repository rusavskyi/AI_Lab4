using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.Features2D;
using Emgu.CV.LineDescriptor;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;


namespace AI_Lab4
{
    class Program
    {
        static void Main(string[] args)
        {
            VectorOfKeyPoint modelKeyPoints, observedKeyPoints = new VectorOfKeyPoint();
            Mat img0 = new Mat(@"D:\Desktop\SI4\testImg0.jpg");
            Mat img1 = new Mat(@"D:\Desktop\SI4\testImg1.jpg");
            VectorOfVectorOfDMatch vectorOfDMatch = new VectorOfVectorOfDMatch();
            FastDetector fd = new FastDetector();
            MKeyPoint[] points0 = fd.Detect(img0);
            MKeyPoint[] points1 = fd.Detect(img1);
            Console.WriteLine(points0.Length);
            Console.WriteLine(points1.Length);
            MKeyPoint[] points2 = fd.Detect(img1, img0);
            Console.WriteLine(points2.Length);
            SIFT sift = new SIFT();
            GpuMat outputArray = new GpuMat();
            sift.DetectAndCompute(img0, img1, new VectorOfKeyPoint(points2), outputArray, false);

        }
    }
}
