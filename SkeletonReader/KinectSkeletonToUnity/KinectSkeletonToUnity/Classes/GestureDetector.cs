using System;
using System.Collections.Generic;
using K = Microsoft.Kinect;

namespace KinectExporterFast.Classes
{
    public static class GestureDetector
    {
        // Historia X dla prawej ręki – na użytkownika
        private static readonly Dictionary<ulong, List<float>> rightHandXHistory =
            new Dictionary<ulong, List<float>>();

        // Czas ostatniego wykrycia machania – na użytkownika (żeby nie spamować)
        private static readonly Dictionary<ulong, DateTime> lastWaveTime =
            new Dictionary<ulong, DateTime>();

        // progi – możesz regulować
        private const float MIN_WAVE_HEIGHT_CM = 10f;      // ręka min. 10 cm powyżej ramion
        private const float MIN_WAVE_AMPLITUDE_CM = 12f;   // machanie na szerokość min. 12 cm
        private const int MIN_HISTORY = 10;                // min. liczba próbek historii

        public static GestureInfo DetectGestureForBody(K.Body body, float positionScale)
        {
            GestureInfo info = new GestureInfo
            {
                HasGesture = false,
                Hand = "",
                Name = "",
                State = "",
                Confidence = 0f
            };

            if (body == null || !body.IsTracked)
                return info;

            var joints = body.Joints;

            var handRight = joints[K.JointType.HandRight];
            var shoulderRight = joints[K.JointType.ShoulderRight];
            var shoulderLeft = joints[K.JointType.ShoulderLeft];

            if (handRight.TrackingState == K.TrackingState.NotTracked ||
                shoulderRight.TrackingState == K.TrackingState.NotTracked ||
                shoulderLeft.TrackingState == K.TrackingState.NotTracked)
            {
                return info;
            }

            // pozycje w cm
            float handX = handRight.Position.X * positionScale;
            float handY = handRight.Position.Y * positionScale;
            float shoulderCenterY = 0.5f *
                (shoulderRight.Position.Y + shoulderLeft.Position.Y) * positionScale;

            // historia X
            if (!rightHandXHistory.TryGetValue(body.TrackingId, out var list))
            {
                list = new List<float>(30);
                rightHandXHistory[body.TrackingId] = list;
            }

            list.Add(handX);
            if (list.Count > 30) list.RemoveAt(0); // max 30 próbek (~1 sekunda przy 30 fps)

            // warunek: ręka powyżej ramion
            if (handY > shoulderCenterY + MIN_WAVE_HEIGHT_CM && list.Count >= MIN_HISTORY)
            {
                float minX = float.MaxValue;
                float maxX = float.MinValue;

                for (int i = 0; i < list.Count; i++)
                {
                    float x = list[i];
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                }

                float amplitude = maxX - minX; // w cm

                if (amplitude > MIN_WAVE_AMPLITUDE_CM)
                {
                    var now = DateTime.UtcNow;

                    if (!lastWaveTime.TryGetValue(body.TrackingId, out var last) ||
                        (now - last).TotalSeconds > 0.7) // min przerwa między „wave”
                    {
                        lastWaveTime[body.TrackingId] = now;

                        info.HasGesture = true;
                        info.Hand = "right";
                        info.Name = "wave";
                        info.State = "start";
                        info.Confidence = Math.Min(1f, amplitude / 30f); // im większa amplituda, tym wyższa pewność

                        return info;
                    }
                }
            }

            // jeśli nic nie wykryliśmy:
            return info;
        }
    }
}
