using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using Emgu.CV;
using Emgu.CV.Structure;

using CvImageType = Emgu.CV.Image<Emgu.CV.Structure.Rgb, System.Byte>;

namespace ImageCut {

    public static class Program {

        private static int CUT_SIZE = 64;
        private static int BOUNDING_BOX_STEP = 2;

        private static StreamWriter _labelWriter;
        private static List<BoundingBox> _boundingBoxes;
        private static string BASE_DIR = @"C:\Users\Paschl\Documents\GitHub\UdacityFinalProject\ros\src\tl_detector\light_classification\training\";
        private static string SOURCE_DIR = BASE_DIR + "source/";
        private static string PROCESSED_DIR = BASE_DIR + "processed/";

        public static void Main(string[] args) {
            //CvImageType cvImg = new Image<Rgb, Byte>((Bitmap)Image.FromFile(SOURCE_DIR + "images/image_0.jpg"));
            //CvImageType cvImgNew = cvImg.SmoothGaussian(5);
            //CvImageType cvImgNew = cvImg.Erode(1);
            //CvImageType[] cvImgNew = cvImg * 1.1;
            //cvImgNew.Save(BASE_DIR + "output.jpg");
            //return;

            _boundingBoxes = CreateSplitBoxes();
            string[] imageFiles = Directory.GetFiles(SOURCE_DIR + "images", "*.jpg", SearchOption.TopDirectoryOnly);
            string labelFile = PROCESSED_DIR + "labels.txt";
            _labelWriter = new StreamWriter(labelFile);

            int imageIdx = 0;
            foreach (string imageFile in imageFiles) {
                string fileName = Path.GetFileNameWithoutExtension(imageFile);
                string labelFileName = SOURCE_DIR + "labels/" + fileName + ".xml";
                Image image = Image.FromFile(imageFile);

                int cutStartIdx = 0;
                BoundingBox bbox = null;

                if (File.Exists(labelFileName)) {
                    Tuple<BoundingBox, string> label = ParseLabelFile(labelFileName);
                    bbox = label.Item1;
                    Image img = CropImage(image, bbox);
                    cutStartIdx = CreateAndSaveImageVariations(img, imageIdx, label.Item2);
                }

                SplitAndSaveImage(image, imageIdx, cutStartIdx, bbox);
                imageIdx++;
            }

            _labelWriter.Flush();
            _labelWriter.Close();
        }

        private static void AddLabel(string label, string imagePath) {
            _labelWriter.WriteLine(label + ";" + imagePath.Replace('\\', '/'));
        }

        private static int CreateAndSaveImageVariations(Image img, int imageIdx, string label) {
            int subImageIdx = 0;
            img = ResizeImage(img, CUT_SIZE, CUT_SIZE);
            CvImageType cvImg = new Image<Rgb, Byte>((Bitmap)img);
            CvImageType[] cvImgNew = {
                cvImg,
                cvImg.SmoothGaussian(5),
                cvImg.SmoothGaussian(7),
                cvImg.Erode(1),
                cvImg.Erode(2)
            };

            foreach (CvImageType image in cvImgNew) {
                string imagePath = PROCESSED_DIR + "images/" + imageIdx + "-" + subImageIdx + ".jpg";
                AddLabel(label, imagePath);
                image.Save(imagePath);
                subImageIdx++;
            }

            return subImageIdx;
        }

        private static List<BoundingBox> CreateSplitBoxes() {
            int width = 800;
            int height = 600;
            List<BoundingBox> bboxes = new List<BoundingBox>();

            for (int y = 0; y <= height - CUT_SIZE; y += BOUNDING_BOX_STEP * CUT_SIZE) {
                for (int x = 0; x <= width - CUT_SIZE; x += BOUNDING_BOX_STEP * CUT_SIZE) {
                    bboxes.Add(new BoundingBox(x, y, CUT_SIZE, CUT_SIZE));
                }
            }

            return bboxes;
        }

        private static void SplitAndSaveImage(Image image, int imageIdx, int cutStartIdx, BoundingBox labelBbox) {
            int cutIdx = cutStartIdx;

            foreach (BoundingBox bbox in _boundingBoxes) {
                if (labelBbox != null && DoBoxesIntersect(bbox, labelBbox)) {
                    continue;
                }

                Image cutImage = CropImage(image, bbox);
                string imagePath = PROCESSED_DIR + "images/" + imageIdx + "-" + cutIdx + ".jpg";
                AddLabel("0", imagePath);

                cutImage.Save(imagePath, ImageFormat.Jpeg);
                cutIdx++;
            }
        }

        private static Image CropImage(Image image, BoundingBox bbox) {
            Rectangle imageRect = new Rectangle(0, 0, bbox.Width, bbox.Height);
            Bitmap croppedImage = new Bitmap(bbox.Width, bbox.Height);
            Rectangle cropRect = new Rectangle(bbox.X, bbox.Y, bbox.Width, bbox.Height);

            using (Graphics gfx = Graphics.FromImage(croppedImage))
            {
                gfx.DrawImage(image, imageRect, cropRect, GraphicsUnit.Pixel);
            }

            return croppedImage;
        }

        private static bool DoBoxesIntersect(BoundingBox a, BoundingBox b) {
            return (Math.Abs(a.X - b.X) * 2 < (a.Width + b.Width)) &&
                   (Math.Abs(a.Y - b.Y) * 2 < (a.Height + b.Height));
        }

        private static Tuple<BoundingBox, string> ParseLabelFile(string fileName) {
            int minX = 0;
            int minY = 0;
            int maxX = 0;
            int maxY = 0;
            string label = null;

            using (StreamReader reader = new StreamReader(fileName)) {
                while (!reader.EndOfStream) {
                    string line = reader.ReadLine().Trim();
                    if (line.StartsWith("<xmin>")) {
                        minX = Convert.ToInt32(line.Replace("<xmin>", "").Replace("</xmin>", ""));
                    }
                    else if (line.StartsWith("<xmax>"))
                    {
                        maxX = Convert.ToInt32(line.Replace("<xmax>", "").Replace("</xmax>", ""));
                    }
                    else if (line.StartsWith("<ymin>"))
                    {
                        minY = Convert.ToInt32(line.Replace("<ymin>", "").Replace("</ymin>", ""));
                    }
                    else if (line.StartsWith("<ymax>"))
                    {
                        maxY = Convert.ToInt32(line.Replace("<ymax>", "").Replace("</ymax>", ""));
                    }
                    else if (line.StartsWith("<name>"))
                    {
                        label = line.Replace("<name>", "").Replace("</name>", "");
                    }
                }
            }

            BoundingBox bbox = new BoundingBox(minX, minY, maxX - minX, maxY - minY);
            if (label == "Green") {
                label = "1";
            }
            else if (label == "Yellow")
            {
                label = "2";
            }
            else if (label == "Red")
            {
                label = "3";
            }

            Tuple<BoundingBox, string> result = new Tuple<BoundingBox, string>(bbox, label);
            return result;
        }

        private static Bitmap ResizeImage(Image image, int width, int height) {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage)) {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes()) {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}
