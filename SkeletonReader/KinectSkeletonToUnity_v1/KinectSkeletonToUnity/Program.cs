using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using System.Drawing;
using System.Diagnostics;
using K = Microsoft.Kinect;

namespace KinectExporterFast
{
    class Program
    {
        // ====== KONFIG ======
        enum OutputMode { JSON = 1, UDP = 2, Both = 3 }

        static OutputMode OUTPUT = OutputMode.Both;

        // Plik wynikowy: jedna tablica JSON:
        // [
        //   { "Data": [ {userid,pose}, ... ], "DelayMs": 0 },
        //   { "Data": [ {userid,pose}, ... ], "DelayMs": 33 },
        //   ...
        // ]
        static string JSON_PATH = "kinect_packets.json";

        static string UDP_HOST = "127.0.0.1";
        static int UDP_PORT = 7001;

        static int MAX_PEOPLE = 6;
        static int MAX_FPS_SEND = 30;  // maks. pakietów na sekundę

        // skala i układ (metry -> cm, Z odwrócone jak wcześniej)
        static float POSITION_SCALE = 100.0f;
        static bool INVERT_Z = true;

        // ====== STAN ======
        static K.KinectSensor sensor;
        static K.BodyFrameReader bodyReader;
        static K.Body[] bodies;

        // log do pliku: Data (stary JSON) + DelayMs
        struct LogItem
        {
            public string Data;   // stary JSON: [ {userid,pose}, ... ]
            public long DelayMs;
        }

        static BlockingCollection<LogItem> fileQueue = new BlockingCollection<LogItem>(new ConcurrentQueue<LogItem>());
        static BlockingCollection<string> udpQueue = new BlockingCollection<string>(new ConcurrentQueue<string>());
        static Thread fileThread, udpThread;

        // podgląd
        static PreviewForm preview;
        static Thread previewThread;

        // throttling FPS
        static readonly Stopwatch sw = Stopwatch.StartNew();
        static long lastTickMs = 0;   // do ograniczania FPS
        static long lastEmitMs = 0;   // do liczenia DelayMs

        // prealokacje do JSON-a
        static readonly K.JointType[] SINGLE_JOINTS = new K.JointType[]
        {
            K.JointType.Head, K.JointType.ShoulderLeft, K.JointType.ShoulderRight,
            K.JointType.ElbowLeft, K.JointType.ElbowRight, K.JointType.WristLeft, K.JointType.WristRight,
            K.JointType.HipLeft, K.JointType.HipRight, K.JointType.KneeLeft, K.JointType.KneeRight,
            K.JointType.AnkleLeft, K.JointType.AnkleRight
        };
        static readonly string[] SINGLE_NAMES = new string[]
        {
            "Head","LeftClavicle","RightClavicle",
            "LeftElbow","RightElbow","LeftWrist","RightWrist",
            "LeftHip","RightHip","LeftKnee","RightKnee",
            "LeftAnkle","RightAnkle"
        };
        static readonly (K.JointType a, K.JointType b, string name)[] MID_JOINTS = new (K.JointType, K.JointType, string)[]
        {
            (K.JointType.ShoulderLeft, K.JointType.ShoulderRight, "Neck"),
            (K.JointType.HipLeft,      K.JointType.HipRight,      "Pelvis"),
        };

        static readonly StringBuilder sb = new StringBuilder(4096);
        static volatile bool _running = true;

        // ====== MAIN ======
        [STAThread]
        static void Main(string[] args)
        {
            Console.WriteLine("KinectExporter FAST (JSON z Data+DelayMs, NET 4.7.2 x64)");

            // okno podglądu (oddzielny STA)
            previewThread = new Thread(() =>
            {
                preview = new PreviewForm();
                Application.Run(preview);
            });
            previewThread.IsBackground = true;
            previewThread.SetApartmentState(ApartmentState.STA);
            previewThread.Start();

            // plik wynikowy – upewniamy się, że jest "[]"
            EnsureJsonArrayFile(JSON_PATH);

            // wątki IO
            fileThread = new Thread(FileWorker) { IsBackground = true, Priority = ThreadPriority.BelowNormal };
            udpThread = new Thread(UdpWorker) { IsBackground = true, Priority = ThreadPriority.BelowNormal };
            fileThread.Start();
            udpThread.Start();

            // Kinect
            sensor = K.KinectSensor.GetDefault();
            if (sensor == null)
            {
                Console.Error.WriteLine("Brak KinectSensor (SDK 2.0).");
                return;
            }

            bodyReader = sensor.BodyFrameSource.OpenReader();
            bodies = new K.Body[sensor.BodyFrameSource.BodyCount];
            bodyReader.FrameArrived += BodyReader_FrameArrived;

            sensor.Open();

            Console.CancelKeyPress += (s, e) => { e.Cancel = true; _running = false; };

            Console.WriteLine("Running. Esc/Ctrl+C aby zakończyć.");
            while (_running)
            {
                if (Console.KeyAvailable)
                {
                    var k = Console.ReadKey(true).Key;
                    if (k == ConsoleKey.Escape) _running = false;
                }
                Thread.Sleep(20);
            }

            // cleanup
            try { bodyReader.FrameArrived -= BodyReader_FrameArrived; } catch { }
            try { bodyReader?.Dispose(); } catch { }
            try { if (sensor?.IsOpen == true) sensor.Close(); } catch { }

            fileQueue.CompleteAdding();
            udpQueue.CompleteAdding();
            fileThread.Join();
            udpThread.Join();

            Console.WriteLine("Bye.");
        }

        // ====== BODY EVENT (główna pętla danych) ======
        private static void BodyReader_FrameArrived(object sender, K.BodyFrameArrivedEventArgs e)
        {
            var frame = e.FrameReference.AcquireFrame();
            if (frame == null) return;

            using (frame)
            {
                frame.GetAndRefreshBodyData(bodies);
            }

            // throttling – max MAX_FPS_SEND pakietów/s
            long now = sw.ElapsedMilliseconds;
            long minDelta = 1000 / Math.Max(1, MAX_FPS_SEND);
            if (now - lastTickMs < minDelta) return;
            lastTickMs = now;

            // zbuduj "stary" JSON: [ {userid, pose}, ... ]
            var json = BuildUsersArrayJson(bodies);
            if (json == null) return;

            // oblicz opóźnienie od poprzedniego wysłanego pakietu
            long delay = (lastEmitMs == 0) ? 0 : (now - lastEmitMs);
            lastEmitMs = now;

            // do pliku: nowa struktura { "Data": <stary json>, "DelayMs": <ms> }
            if (OUTPUT == OutputMode.JSON || OUTPUT == OutputMode.Both)
                fileQueue.Add(new LogItem { Data = json, DelayMs = delay });

            // UDP: wysyłamy sam "stary" json (tablica użytkowników)
            if (OUTPUT == OutputMode.UDP || OUTPUT == OutputMode.Both)
                udpQueue.Add(json);

            // Podgląd 2D
            if (preview != null && !preview.IsDisposed)
            {
                var snapshot = PreviewForm.BuildSnapshot(bodies, POSITION_SCALE);
                preview.SetSkeletons(snapshot);
            }
        }

        // ====== Budowa starego JSON-a: [ {userid, pose}, ... ] ======
        static string BuildUsersArrayJson(K.Body[] bs)
        {
            sb.Clear();
            sb.Append('[');
            int countUsers = 0;

            for (int i = 0; i < bs.Length && countUsers < MAX_PEOPLE; i++)
            {
                var b = bs[i];
                if (b == null || !b.IsTracked) continue;

                var j = b.Joints;
                if (!Tracked(j[K.JointType.Head]) ||
                    !Tracked(j[K.JointType.ShoulderLeft]) ||
                    !Tracked(j[K.JointType.ShoulderRight]) ||
                    !Tracked(j[K.JointType.HipLeft]) ||
                    !Tracked(j[K.JointType.HipRight])) continue;

                if (countUsers++ > 0) sb.Append(',');

                sb.Append("{\"userid\":\"");
                sb.Append(b.TrackingId);
                sb.Append("\",\"pose\":{");

                // single
                for (int s = 0; s < SINGLE_JOINTS.Length; s++)
                {
                    var jt = SINGLE_JOINTS[s];
                    var name = SINGLE_NAMES[s];
                    var jj = j[jt];
                    if (!Tracked(jj)) continue;
                    WriteVec3(name, jj.Position);
                }

                // mids
                for (int m = 0; m < MID_JOINTS.Length; m++)
                {
                    var a = j[MID_JOINTS[m].a];
                    var b2 = j[MID_JOINTS[m].b];
                    if (!Tracked(a) || !Tracked(b2)) continue;

                    K.CameraSpacePoint p;
                    p.X = 0.5f * (a.Position.X + b2.Position.X);
                    p.Y = 0.5f * (a.Position.Y + b2.Position.Y);
                    p.Z = 0.5f * (a.Position.Z + b2.Position.Z);
                    WriteVec3(MID_JOINTS[m].name, p);
                }

                sb.Append('}');
                sb.Append('}');
            }

            sb.Append(']');
            if (countUsers == 0) return null;
            return sb.ToString();

            void WriteVec3(string key, K.CameraSpacePoint p)
            {
                float x = p.X * POSITION_SCALE;
                float y = p.Y * POSITION_SCALE;
                float z = p.Z * POSITION_SCALE;
                if (INVERT_Z) z = -z;

                if (sb[sb.Length - 1] != '{') sb.Append(',');

                sb.Append('\"').Append(key).Append("\":[");
                sb.Append(Num(x)).Append(',').Append(Num(y)).Append(',').Append(Num(z)).Append(']');
            }

            string Num(float f) => f.ToString("G9", System.Globalization.CultureInfo.InvariantCulture);
        }

        static bool Tracked(K.Joint j) =>
            j.TrackingState == K.TrackingState.Tracked || j.TrackingState == K.TrackingState.Inferred;

        // ====== FileWorker – dopisywanie do tablicy JSON ======

        static string JsonStringEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            var sb = new StringBuilder(s.Length + 16);

            foreach (var c in s)
            {
                switch (c)
                {
                    case '\\': sb.Append("\\\\"); break;
                    case '"': sb.Append("\\\""); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 32)
                            sb.Append("\\u").Append(((int)c).ToString("X4"));
                        else
                            sb.Append(c);
                        break;
                }
            }
            return sb.ToString();
        }


        static void FileWorker()
        {
            // Plik ma strukturę:
            // [ { "Data": [...], "DelayMs": 0 }, { "Data": [...], "DelayMs": 33 }, ... ]

            foreach (var item in fileQueue.GetConsumingEnumerable())
            {
                try
                {
                    string txt = File.Exists(JSON_PATH) ? File.ReadAllText(JSON_PATH, Encoding.UTF8).Trim() : "[]";
                    if (string.IsNullOrWhiteSpace(txt)) txt = "[]";

                    // "Data" to stary JSON (tablica) – wstawiamy BEZ cudzysłowu
                    // "DelayMs" jako liczba
                    string dataEsc = JsonStringEscape(item.Data);
                    string entry = $"{{\"Data\":\"{dataEsc}\",\"DelayMs\":{item.DelayMs}}}";

                    if (txt == "[]")
                        txt = "[" + entry + "]";
                    else if (txt.EndsWith("]"))
                        txt = txt.Substring(0, txt.Length - 1) + "," + entry + "]";
                    else
                        txt = "[" + entry + "]";

                    File.WriteAllText(JSON_PATH, txt, Encoding.UTF8);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine("FILE: " + ex.Message);
                }
            }
        }

        // ====== UDP Worker ======
        static void UdpWorker()
        {
            using (var udp = new UdpClient())
            {
                udp.Connect(UDP_HOST, UDP_PORT);
                foreach (var payload in udpQueue.GetConsumingEnumerable())
                {
                    var bytes = Encoding.UTF8.GetBytes(payload);
                    try { udp.Send(bytes, bytes.Length); }
                    catch { /* pomijamy chwilowe błędy */ }
                }
            }
        }

        static void EnsureJsonArrayFile(string path)
        {
            try
            {
                if (!File.Exists(path)) File.WriteAllText(path, "[]", Encoding.UTF8);
            }
            catch { }
        }
    }

    // ================== OKNO PODGLĄDU ==================
    public class PreviewForm : Form
    {
        public struct Pt { public float X, Y; }
        public class Skeleton2D
        {
            public ulong Id;
            public readonly Dictionary<string, Pt> P = new Dictionary<string, Pt>(24);
        }

        private readonly object _lock = new object();
        private List<Skeleton2D> _skeletons = new List<Skeleton2D>();

        private float pxPerCm = 4f;
        private readonly System.Windows.Forms.Timer timer;

        private readonly Pen pen = new Pen(Color.Lime, 3f);
        private readonly Pen penAxis = new Pen(Color.FromArgb(120, 255, 255, 255), 2f);
        private readonly Pen penGrid = new Pen(Color.FromArgb(40, 255, 255, 255), 1f);
        private readonly Brush br = new SolidBrush(Color.Orange);
        private readonly Brush brText = new SolidBrush(Color.White);
        private readonly Font font = new Font("Consolas", 10f);

        private static readonly (string a, string b)[] Bones = new (string, string)[]
        {
            ("Head","Neck"),
            ("Neck","LeftClavicle"), ("LeftClavicle","LeftElbow"), ("LeftElbow","LeftWrist"),
            ("Neck","RightClavicle"), ("RightClavicle","RightElbow"), ("RightElbow","RightWrist"),
            ("Pelvis","LeftHip"), ("LeftHip","LeftKnee"), ("LeftKnee","LeftAnkle"),
            ("Pelvis","RightHip"), ("RightHip","RightKnee"), ("RightKnee","RightAnkle"),
            ("Neck","Pelvis")
        };

        public PreviewForm()
        {
            Text = "Kinect Preview (Data+DelayMs JSON)  [+/- zoom, R reset]";
            Width = 900; Height = 700;
            DoubleBuffered = true; BackColor = Color.Black;

            timer = new System.Windows.Forms.Timer { Interval = 33 };
            timer.Tick += (s, e) => Invalidate();
            timer.Start();

            KeyDown += (s, e) =>
            {
                if (e.KeyCode == Keys.Add || e.KeyCode == Keys.Oemplus) { pxPerCm *= 1.1f; Invalidate(); }
                if (e.KeyCode == Keys.Subtract || e.KeyCode == Keys.OemMinus)
                {
                    pxPerCm /= 1.1f; if (pxPerCm < 0.5f) pxPerCm = 0.5f; Invalidate();
                }
                if (e.KeyCode == Keys.R) { pxPerCm = 4f; Invalidate(); }
            };
        }

        public void SetSkeletons(List<Skeleton2D> list)
        {
            lock (_lock)
            {
                _skeletons = list ?? new List<Skeleton2D>();
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighSpeed;

            float cx = ClientSize.Width / 2f;
            float cy = ClientSize.Height / 2f;

            DrawGrid(g, cx, cy);

            List<Skeleton2D> local;
            lock (_lock) local = _skeletons;

            foreach (var sk in local)
            {
                foreach (var b in Bones)
                {
                    if (!sk.P.TryGetValue(b.a, out var A)) continue;
                    if (!sk.P.TryGetValue(b.b, out var B)) continue;

                    var p1 = new PointF(cx + A.X * pxPerCm, cy - A.Y * pxPerCm);
                    var p2 = new PointF(cx + B.X * pxPerCm, cy - B.Y * pxPerCm);
                    g.DrawLine(pen, p1, p2);
                }

                foreach (var kv in sk.P)
                {
                    var p = new PointF(cx + kv.Value.X * pxPerCm, cy - kv.Value.Y * pxPerCm);
                    g.FillEllipse(br, p.X - 4, p.Y - 4, 8, 8);
                }
            }

            g.DrawString($"Scale: {pxPerCm:0.0} px/cm   (~30 FPS timer)", font, brText, 6, ClientSize.Height - 22);
        }

        private void DrawGrid(Graphics g, float cx, float cy)
        {
            int stepCm = 10;
            int maxX = (int)Math.Ceiling((ClientSize.Width / 2f) / pxPerCm / stepCm) * stepCm;
            int maxY = (int)Math.Ceiling((ClientSize.Height / 2f) / pxPerCm / stepCm) * stepCm;

            for (int x = -maxX; x <= maxX; x += stepCm)
            {
                float px = cx + x * pxPerCm;
                g.DrawLine(penGrid, px, 0, px, ClientSize.Height);
            }
            for (int y = -maxY; y <= maxY; y += stepCm)
            {
                float py = cy - y * pxPerCm;
                g.DrawLine(penGrid, 0, py, ClientSize.Width, py);
            }

            g.DrawLine(penAxis, 0, cy, ClientSize.Width, cy);
            g.DrawLine(penAxis, cx, 0, cx, ClientSize.Height);
        }

        // budowanie snapshotu do podglądu
        public static List<Skeleton2D> BuildSnapshot(K.Body[] bodies, float positionScale)
        {
            var list = new List<Skeleton2D>(6);
            foreach (var body in bodies)
            {
                if (body == null || !body.IsTracked) continue;

                var j = body.Joints;
                var sk = new Skeleton2D { Id = body.TrackingId };

                void Add(string name, K.JointType jt)
                {
                    var jj = j[jt];
                    if (jj.TrackingState == K.TrackingState.NotTracked) return;
                    float x = jj.Position.X * positionScale;
                    float y = jj.Position.Y * positionScale;
                    sk.P[name] = new Pt { X = x, Y = y };
                }
                void AddMid(string name, K.JointType a, K.JointType b)
                {
                    var ja = j[a]; var jb = j[b];
                    if (ja.TrackingState == K.TrackingState.NotTracked || jb.TrackingState == K.TrackingState.NotTracked) return;
                    float x = 0.5f * (ja.Position.X + jb.Position.X) * positionScale;
                    float y = 0.5f * (ja.Position.Y + jb.Position.Y) * positionScale;
                    sk.P[name] = new Pt { X = x, Y = y };
                }

                Add("Head", K.JointType.Head);
                AddMid("Neck", K.JointType.ShoulderLeft, K.JointType.ShoulderRight);
                AddMid("Pelvis", K.JointType.HipLeft, K.JointType.HipRight);

                Add("LeftClavicle", K.JointType.ShoulderLeft);
                Add("RightClavicle", K.JointType.ShoulderRight);
                Add("LeftElbow", K.JointType.ElbowLeft);
                Add("RightElbow", K.JointType.ElbowRight);
                Add("LeftWrist", K.JointType.WristLeft);
                Add("RightWrist", K.JointType.WristRight);
                Add("LeftHip", K.JointType.HipLeft);
                Add("RightHip", K.JointType.HipRight);
                Add("LeftKnee", K.JointType.KneeLeft);
                Add("RightKnee", K.JointType.KneeRight);
                Add("LeftAnkle", K.JointType.AnkleLeft);
                Add("RightAnkle", K.JointType.AnkleRight);

                list.Add(sk);
            }
            return list;
        }
    }
}
