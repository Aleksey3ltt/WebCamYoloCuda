using System.Data;
using System.Diagnostics.Metrics;
using System.Windows.Forms;
using AForge.Video.DirectShow;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Reflection.Emit;
using Emgu.CV.CvEnum;
using System.Collections.Generic;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.ToolBar;

namespace WebCamYoloCuda
{
    public partial class Form1 : Form
    {
        private FilterInfoCollection CaptureDevices;
        private VideoCaptureDevice videoSource;
        int setResolution;
        string netName;
        string pathNet;
        Net Model;
        double fps = 0;
        string Pathconfig;
        string PathWeights;
        string[] pathClassNames;
        List<string> readFileNames;
        VectorOfMat layerOutputs = new VectorOfMat();
        List<(string, float, int, int, int, int)> cnnPredictions;
        (string, float, int, int, int, int) resultCNN = ("0", 0, 0, 0, 0, 0);

        //public float ConfidenceThreshold { get; set; }
        //public float NMSThreshold { get; set; }
        
        public Form1()
        {
            InitializeComponent();
            CudaChek();
            comboBox2.SelectedIndex = 0;
            comboBox2.SelectedIndexChanged += comboBox2_SelectedIndexChanged;
            label1.Text = null;
            comboBox3.SelectedIndex = 0;
            comboBox3.SelectedIndexChanged += comboBox3_SelectedIndexChanged;
            radioButton1.Checked = true;
        }

        public void CudaChek()
        {
            if (CudaInvoke.HasCuda)
            {
                textBox1.Text = CudaInvoke.GetCudaDevicesSummary();
            }
            else
            {
                textBox1.Text = "No CUDA";
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            CaptureDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            foreach (FilterInfo Device in CaptureDevices)
            {
                comboBox1.Items.Add(Device.Name);
            }
            comboBox1.SelectedIndex = 0;
        }

        private void videoSource_NewFrame(object sender, AForge.Video.NewFrameEventArgs eventArgs)
        {
            Bitmap img = (Bitmap)eventArgs.Frame.Clone();
            int width = img.Width;
            int height = img.Height;
            Image<Bgr, byte> source = img.ToImage<Bgr, byte>();
            CNN_detection(source, width, height);
            AddDetailsToPictureBox(source, cnnPredictions);
            pictureBox1.Image = source.AsBitmap();
        }

        private void CNN_detection(Image<Bgr, byte> source, int width, int height)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            Image<Bgr, byte> resizedImage = null;

            if (radioButton1.Checked == true)
            {
                resizedImage = source.Resize(416, 416, Inter.Linear);
            }
            if (radioButton2.Checked == true)
            {
                resizedImage = source;
            }

            Mat inputBlob = DnnInvoke.BlobFromImage(resizedImage, 1 / 255.0, swapRB: true);
            Model.SetInput(inputBlob);
            Model.Forward(layerOutputs, Model.UnconnectedOutLayersNames);
            List<Rectangle> boxes = new List<Rectangle>();
            List<float> confidences = new List<float>();
            cnnPredictions = new List<(string, float, int, int, int, int)>();
            
            float ConfidenceThreshold = 0.3f;
            float NMSThreshold = 0.5f;
            
            for (int k = 0; k < layerOutputs.Size; k++)
            {
                float[,] lo = (float[,])layerOutputs[k].GetData();
                int len = lo.GetLength(0);
                for (int i = 0; i < len; i++)
                {
                    if (lo[i, 4] < ConfidenceThreshold)
                        continue;
                    float max = 0;
                    int idx = 0;

                    int len2 = lo.GetLength(1);
                    for (int j = 5; j < len2; j++)
                        if (lo[i, j] > max)
                        {
                            max = lo[i, j];
                            idx = j - 5;
                        }

                    if (max > ConfidenceThreshold)
                    {
                        lo[i, 0] *= width;
                        lo[i, 1] *= height;
                        lo[i, 2] *= width;
                        lo[i, 3] *= height;

                        int x = (int)(lo[i, 0] - (lo[i, 2] / 2));
                        int y = (int)(lo[i, 1] - (lo[i, 3] / 2));

                        var rect = new Rectangle(x, y, (int)lo[i, 2], (int)lo[i, 3]);

                        rect.X = rect.X < 0 ? 0 : rect.X;
                        rect.X = rect.X > width ? width - 1 : rect.X;
                        rect.Y = rect.Y < 0 ? 0 : rect.Y;
                        rect.Y = rect.Y > height ? height - 1 : rect.Y;
                        rect.Width = rect.X + rect.Width > width ? width - rect.X - 1 : rect.Width;
                        rect.Height = rect.Y + rect.Height > height ? height - rect.Y - 1 : rect.Height;

                        boxes.Add(rect);
                        confidences.Add(max);
                        resultCNN = (readFileNames[idx], max,  rect.X, rect.Y, rect.Width, rect.Height);
                    }
                }
                int[] bIndexes = DnnInvoke.NMSBoxes(boxes.ToArray(), confidences.ToArray(), ConfidenceThreshold, NMSThreshold);
                if (bIndexes.Length > 0)
                {
                    foreach (var idx in bIndexes)
                    { 
                        cnnPredictions.Add(resultCNN);
                    }
                }
            }
            watch.Stop();
            fps = Math.Round(1 / watch.Elapsed.TotalSeconds, 2);
        }

        public void AddDetailsToPictureBox(Image<Bgr, byte> source, List<(string, float, int, int, int, int)> cnnPredictions)
        {
            CvInvoke.PutText(source, "FPS: " + Convert.ToString(fps), new Point(10, 30), FontFace.HersheyComplex, 0.7, new MCvScalar(0, 0, 255));
            CvInvoke.PutText(source, "CNN: " + netName, new Point(10, 60), FontFace.HersheyComplex, 0.7, new MCvScalar(0, 255, 0));

            foreach (var item in this.cnnPredictions)
            {
                var x = item.Item3;
                var y = item.Item4;
                var width = item.Item5;
                var height = item.Item6;
                float confid = item.Item2;
                string tung = item.Item1;

                var rect = new Rectangle(x, y, width, height);
                CvInvoke.Rectangle(source, rect, new MCvScalar(0, 0, 255), 2);
                CvInvoke.PutText(source, tung+" " + confid, new Point(x, y-5), FontFace.HersheySimplex, 0.8, new MCvScalar(255, 0, 0));
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (videoSource != null && videoSource.IsRunning)
            {
                videoSource.SignalToStop();
                pictureBox1.Image = null;
            }
            Load_CNN_Model();
            videoSource = new VideoCaptureDevice(CaptureDevices[comboBox1.SelectedIndex].MonikerString);
            videoSource.VideoResolution = videoSource.VideoCapabilities[setResolution];
            videoSource.NewFrame += videoSource_NewFrame;
            videoSource.Start();
            label1.Text = videoSource.VideoCapabilities[setResolution].FrameSize.Width + " x " + videoSource.VideoCapabilities[setResolution].FrameSize.Height;
        }

        private void Load_CNN_Model()
        {
            Model = new Net();
            Model = DnnInvoke.ReadNetFromDarknet(Pathconfig, PathWeights);
            //Model.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
            Model.SetPreferableBackend(Emgu.CV.Dnn.Backend.Cuda);
            //Model.SetPreferableTarget(Target.Cuda);
            Model.SetPreferableTarget(Target.Cuda);
        }

        private void Timeout_VideoSource()
        {
            if (videoSource != null && videoSource.IsRunning)
            {
                videoSource.SignalToStop();
                pictureBox1.Image = null;
                Thread.Sleep(500);
                button1_Click(this, new EventArgs());
            }
        }

        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            string selectedResolution = comboBox2.SelectedItem.ToString();
            setResolution = Convert.ToInt32(selectedResolution);
            Timeout_VideoSource();
        }

        private void comboBox3_SelectedIndexChanged(object sender, EventArgs e)
        {
            netName = comboBox3.Text;
            pathNet = Directory.GetCurrentDirectory() + "\\networks\\" + comboBox3.Text + "\\";
            Pathconfig = pathNet + comboBox3.Text + ".cfg";
            PathWeights = pathNet + comboBox3.Text + ".weights";
            pathClassNames = Directory.GetFiles(pathNet, "*.*").Where(f =>f.EndsWith(".names")).ToArray();
            readFileNames = File.ReadAllLines(pathClassNames[0]).ToList();
            Timeout_VideoSource();
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton1.Checked == true)
            {
                comboBox3.Enabled = true;
                Thread.Sleep(200);
            }
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            
            if (radioButton2.Checked == true)
            {
                comboBox3.Enabled = false;
                comboBox3.SelectedIndex = 0;
                Thread.Sleep(200);
            }
        }
    }
}