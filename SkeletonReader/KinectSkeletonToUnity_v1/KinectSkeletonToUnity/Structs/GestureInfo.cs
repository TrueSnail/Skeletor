namespace KinectExporterFast
{
    public struct GestureInfo
    {
        public bool HasGesture;   // czy w tej klatce coś wykryto
        public string Hand;       // "left" / "right"
        public string Name;       // "wave", "thumbs_up", "ok", ...
        public string State;      // "start" / "hold" / "end"
        public float Confidence;  // 0..1
    }

}