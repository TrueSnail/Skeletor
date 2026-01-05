using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SkeletonDataModel
{
    public class Pose
    {
        public double[] Head { get; set; }
        public double[] Neck { get; set; }
        public double[] Pelvis { get; set; }
        public double[] LeftClavicle { get; set; }
        public double[] RightClavicle { get; set; }
        public double[] LeftElbow { get; set; }
        public double[] RightElbow { get; set; }
        public double[] LeftWrist { get; set; }
        public double[] RightWrist { get; set; }
        public double[] LeftHip { get; set; }
        public double[] RightHip { get; set; }
        public double[] LeftKnee { get; set; }
        public double[] RightKnee { get; set; }
        public double[] LeftAnkle { get; set; }
        public double[] RightAnkle { get; set; }
    }

    public class CommandRoot
    {
        public string userid { get; set; }
        public Pose pose { get; set; }
    }

    public class UnityPositionPose
    {
        public Vector3 Head { get; set; }
        public Vector3 Neck { get; set; }
        public Vector3 Pelvis { get; set; }
        public Vector3 LeftClavicle { get; set; }
        public Vector3 RightClavicle { get; set; }
        public Vector3 LeftElbow { get; set; }
        public Vector3 RightElbow { get; set; }
        public Vector3 LeftWrist { get; set; }
        public Vector3 RightWrist { get; set; }
        public Vector3 LeftHip { get; set; }
        public Vector3 RightHip { get; set; }
        public Vector3 LeftKnee { get; set; }
        public Vector3 RightKnee { get; set; }
        public Vector3 LeftAnkle { get; set; }
        public Vector3 RightAnkle { get; set; }

        public static UnityPositionPose FromPose(Pose pose)
        {
            return new UnityPositionPose()
            {
                Head = ListToVector(pose.Head),
                Neck = ListToVector(pose.Neck),
                Pelvis = ListToVector(pose.Pelvis),
                LeftClavicle = ListToVector(pose.LeftClavicle),
                RightClavicle = ListToVector(pose.RightClavicle),
                LeftElbow = ListToVector(pose.LeftElbow),
                RightElbow = ListToVector(pose.RightElbow),
                LeftWrist = ListToVector(pose.LeftWrist),
                RightWrist = ListToVector(pose.RightWrist),
                LeftHip = ListToVector(pose.LeftHip),
                RightHip = ListToVector(pose.RightHip),
                LeftKnee = ListToVector(pose.LeftKnee),
                RightKnee = ListToVector(pose.RightKnee),
                LeftAnkle = ListToVector(pose.LeftAnkle),
                RightAnkle = ListToVector(pose.RightAnkle)
            };
        }

        private static Vector3 ListToVector(double[] list) => new Vector3((float)list[0], (float)list[1], (float)list[2]);
    }

    public class UnityRotationPose
    {
        public Vector3 Position { get; set; }
        public Quaternion Rotation { get; set; }
        public Quaternion LeftUpLeg { get; set; }
        public Quaternion RightUpLeg { get; set; }
        public Quaternion LeftLeg { get; set; }
        public Quaternion RightLeg { get; set; }
        public Quaternion RightArm{ get; set; }
        public Quaternion LeftArm { get; set; }
        public Quaternion RightForeArm { get; set; }
        public Quaternion LeftForeArm { get; set; }
        public Quaternion Neck { get; set; }


        public static UnityRotationPose FromPose(Pose pose)
        {
            var positionPose = UnityPositionPose.FromPose(pose);

            return new UnityRotationPose()
            {
                Position = positionPose.Pelvis,
                Rotation = Quaternion.LookRotation(-Vector3.Cross(positionPose.RightHip - positionPose.LeftHip, Vector3.up), Vector3.Cross(Vector3.Cross(positionPose.RightHip - positionPose.LeftHip, Vector3.up), positionPose.RightHip - positionPose.LeftHip)),
                LeftUpLeg = GetRotation(positionPose.LeftHip, positionPose.LeftKnee, true),
                RightUpLeg = GetRotation(positionPose.RightHip, positionPose.RightKnee, true),
                LeftLeg = GetRotation(positionPose.LeftKnee, positionPose.LeftAnkle, true),
                RightLeg = GetRotation(positionPose.RightKnee, positionPose.RightAnkle, true),
                LeftArm = GetRotation(positionPose.LeftClavicle, positionPose.LeftElbow),
                RightArm = GetRotation(positionPose.RightClavicle, positionPose.RightElbow),
                LeftForeArm = GetRotation(positionPose.LeftElbow, positionPose.LeftWrist),
                RightForeArm = GetRotation(positionPose.RightElbow, positionPose.RightWrist),
                Neck = GetRotation(positionPose.Neck, positionPose.Head)
            };
        }

        private static Quaternion GetRotation(Vector3 startPoint, Vector3 endPoint, bool inverse = false)
        {
            var direction = endPoint - startPoint;
            return Quaternion.LookRotation(Vector3.Cross(direction, Vector3.right) * (inverse ? -1 : 1), direction);
        }
    }
}