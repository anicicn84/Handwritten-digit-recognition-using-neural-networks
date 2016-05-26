using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Drawing;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MasterGUICanvas
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        System.Windows.Point currentPoint = new System.Windows.Point();

        //Button button1 = null;
        //Button button2 = null;
        //Button button3 = null;
        public MainWindow()
        {
            string userName = System.Environment.UserName;
            string folderName = @"C: \Users\" + userName + @"\Desktop\";
            string pathString = System.IO.Path.Combine(folderName, "images");
            System.IO.Directory.CreateDirectory(pathString);
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            //button1 = new Button { Content = "Set", Width = 80, Height = 24 };
            //Canvas.SetLeft(button1, 45);
            //Canvas.SetTop(button1, 325);
            //paintSurface.Children.Add(button1);
            ////button2 = new Button { Content = "Clear" , Width = 80, Height = 24 };
            ////Canvas.SetLeft(button2, 130);
            ////Canvas.SetTop(button2, 325);
            //button2 = new Button();
            //paintSurface.Children.Add(button2);
            //button3 = new Button { Content = "Cancel", Width = 80, Height = 24 };
            //Canvas.SetLeft(button3, 215);
            //Canvas.SetTop(button3, 325);
            //button3.Padding = new Thickness(9, 2, 9, 2);
            //paintSurface.Children.Add(button3);
        }

        public static System.Drawing.Image ScaleImage(System.Drawing.Image image, int maxWidth, int maxHeight)
        {
            var ratioX = (double)maxWidth / image.Width;
            var ratioY = (double)maxHeight / image.Height;
            var ratio = Math.Min(ratioX, ratioY);

            var newWidth = (int)(image.Width * ratio);
            var newHeight = (int)(image.Height * ratio);

            var newImage = new Bitmap(newWidth, newHeight);

            using (var graphics = Graphics.FromImage(newImage))
                graphics.DrawImage(image, 0, 0, newWidth, newHeight);

            return newImage;
        }

        private void Canvas_MouseDown_1(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (e.ButtonState == MouseButtonState.Pressed)
                currentPoint = e.GetPosition(this);
        }
        private void Canvas_MouseMove_1(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                if (currentPoint.X == e.GetPosition(this).X && currentPoint.Y == e.GetPosition(this).Y)
                {
                    return;
                }

                Line line = new Line();

                line.StrokeThickness = 30;
                line.Stroke = System.Windows.Media.Brushes.Black;
                line.StrokeStartLineCap = PenLineCap.Round;
                line.StrokeEndLineCap = PenLineCap.Round;
                line.X1 = currentPoint.X;
                line.Y1 = currentPoint.Y;
                line.X2 = e.GetPosition(this).X;
                line.Y2 = e.GetPosition(this).Y;

                currentPoint = e.GetPosition(this);
                System.Diagnostics.Debug.WriteLine(string.Format("({0}, {1}) - ({2}, {3})", line.X1, line.Y1, line.X2, line.Y2));

                paintSurface.Children.Add(line);
            }
        }

        private void ClearButton_Click(object sender, RoutedEventArgs e)
        {
            paintSurface.Children.Clear();
        }

        public bool ByteArrayToFile(string _FileName, byte[] _ByteArray)
        {
            try
            {
                // Open file for reading
                System.IO.FileStream _FileStream =
                    new System.IO.FileStream(_FileName, System.IO.FileMode.Create,
                                            System.IO.FileAccess.Write);
                // Writes a block of bytes to this stream using data from
                // a byte array.
                _FileStream.Write(_ByteArray, 0, _ByteArray.Length);

                // close file stream
                _FileStream.Close();

                return true;
            }
            catch (Exception _Exception)
            {
                // Error
                Console.WriteLine("Exception caught in process: {0}",
                                    _Exception.ToString());
            }

            // error occured, return false
            return false;
        }

        private void setImagePixelsToBinary(Bitmap imageEx)
        {
            string userName = System.Environment.UserName;
            int x, y;
            int magic_number = 2051;
            int num_of_images = 1;
            int image_height = 28;
            int image_width = 28;
            byte[] imagePixelBytes = new byte[imageEx.Width * imageEx.Height + 16]; //4 * 4 byte values

            //adding magic number to an array of bytes
            imagePixelBytes[0] = (byte)(magic_number >> 24);
            imagePixelBytes[1] = (byte)(magic_number >> 16);
            imagePixelBytes[2] = (byte)(magic_number >> 8);
            imagePixelBytes[3] = (byte)magic_number;

            //adding number of images to an array of bytes
            imagePixelBytes[4] = (byte)(num_of_images >> 24);
            imagePixelBytes[5] = (byte)(num_of_images >> 16);
            imagePixelBytes[6] = (byte)(num_of_images >> 8);
            imagePixelBytes[7] = (byte)num_of_images;

            //adding image height to an array of bytes
            imagePixelBytes[8] = (byte)(image_height >> 24);
            imagePixelBytes[9] = (byte)(image_height >> 16);
            imagePixelBytes[10] = (byte)(image_height >> 8);
            imagePixelBytes[11] = (byte)image_height;

            //adding image width to an array of bytes
            imagePixelBytes[12] = (byte)(image_width >> 24);
            imagePixelBytes[13] = (byte)(image_width >> 16);
            imagePixelBytes[14] = (byte)(image_width >> 8);
            imagePixelBytes[15] = (byte)image_width;

            for (x = 0; x < imageEx.Width; x++)
            {
                for (y = 0; y < imageEx.Height; y++)
                {
                    System.Drawing.Color pixelColor = imageEx.GetPixel(x, y);
                    if (pixelColor.R == 255 && pixelColor.G == 255 && pixelColor.B == 255)
                        imagePixelBytes[28 * y + x + 16] = 0;
                    else
                        imagePixelBytes[28 * y + x + 16] = 255;
                }
            }
            ByteArrayToFile(@"C: \Users\" + userName + @"\Desktop\images\myFile.idx3-ubyte", imagePixelBytes);
        }
        private void SetButton_Click(object sender, RoutedEventArgs e)
        {
            RenderTargetBitmap renderBitmap = new RenderTargetBitmap(
        (int)paintSurface.RenderSize.Width, (int)paintSurface.RenderSize.Height,
        96d, 96d, System.Windows.Media.PixelFormats.Default);
            // needed otherwise the image output is black
            paintSurface.Measure(new System.Windows.Size((int)paintSurface.RenderSize.Width, (int)paintSurface.RenderSize.Height));
            paintSurface.Arrange(new Rect(new System.Windows.Size((int)paintSurface.RenderSize.Width, (int)paintSurface.RenderSize.Height)));

            renderBitmap.Render(paintSurface);

            //BitmapEncoder encoder = new JpegBitmapEncoder();
            PngBitmapEncoder encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(renderBitmap));
            string userName = System.Environment.UserName;

            using (System.IO.FileStream file = System.IO.File.Create(@"C:\Users\"+userName+@"\Desktop\images\image.png"))
            {
                encoder.Save(file);
            }

            using (var image = System.Drawing.Image.FromFile(@"C:\Users\" + userName + @"\Desktop\images\image.png"))
            using (var newImage = ScaleImage(image, 28, 28))
            {
                using (var b = new Bitmap(newImage.Width, newImage.Height))
                {
                    b.SetResolution(newImage.HorizontalResolution, newImage.VerticalResolution);

                    using (var g = Graphics.FromImage(b))
                    {
                        g.Clear(System.Drawing.Color.White);
                        g.DrawImageUnscaled(newImage, 0, 0);
                    }

                    Bitmap tempImage = new Bitmap(b);
                    //b.Dispose();
                    //newImage.Dispose();

                    Bitmap d;
                    int x, y;

                    // Loop through the images pixels to reset color.
                    for (x = 0; x < tempImage.Width; x++)
                    {
                        for (y = 0; y < tempImage.Height; y++)
                        {
                            System.Drawing.Color pixelColor = tempImage.GetPixel(x, y);
                            int rgb = (int)((pixelColor.R + pixelColor.G + pixelColor.B) / 3);
                            System.Drawing.Color newColor = System.Drawing.Color.FromArgb(rgb, rgb, rgb);
                            tempImage.SetPixel(x, y, newColor); // Now greyscale
                        }
                    }
                    d = tempImage;   // d is grayscale version of c 

                    d.Save(@"C:\Users\" + userName + @"\Desktop\images\imageResized.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);

                    //Byte[] imageBytesArray = (Byte[])new ImageConverter().ConvertTo(d, typeof(Byte[]));

                    //System.Drawing.Image newImage1 = new Bitmap(d);
                    //newImage1 = b;
                    setImagePixelsToBinary(d);

                    tempImage.Dispose();
                }
            }
        }
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();

            string userName = System.Environment.UserName;

            System.IO.DirectoryInfo di = new System.IO.DirectoryInfo(@"C: \Users\" + userName + @"\Desktop\images");

            foreach (System.IO.FileInfo file in di.GetFiles())
            {
                file.Delete();
            }
            foreach (System.IO.DirectoryInfo dir in di.GetDirectories())
            {
                dir.Delete(true);
            }
        }
    }
}
