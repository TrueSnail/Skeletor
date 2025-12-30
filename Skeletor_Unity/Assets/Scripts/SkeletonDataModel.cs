using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace SkeletonDataModel
{
    public class Bones
    {
        public List<double> Wrist { get; set; }
        public List<double> Thumb_CMC { get; set; }
        public List<double> Thumb_MCP { get; set; }
        public List<double> Thumb_IP { get; set; }
        public List<double> Thumb_Tip { get; set; }
        public List<double> Index_MCP { get; set; }
        public List<double> Index_PIP { get; set; }
        public List<double> Index_DIP { get; set; }
        public List<double> Index_Tip { get; set; }
        public List<double> Middle_MCP { get; set; }
        public List<double> Middle_PIP { get; set; }
        public List<double> Middle_DIP { get; set; }
        public List<double> Middle_Tip { get; set; }
        public List<double> Ring_MCP { get; set; }
        public List<double> Ring_PIP { get; set; }
        public List<double> Ring_DIP { get; set; }
        public List<double> Ring_Tip { get; set; }
        public List<double> Pinky_MCP { get; set; }
        public List<double> Pinky_PIP { get; set; }
        public List<double> Pinky_DIP { get; set; }
        public List<double> Pinky_Tip { get; set; }
    }

    public class Hand
    {
        public string hand { get; set; }
        public Bones bones { get; set; }
    }

    public class Pose
    {
        public List<double> Head { get; set; }
        public List<double> Neck { get; set; }
        public List<double> Pelvis { get; set; }
        public List<double> LeftClavicle { get; set; }
        public List<double> RightClavicle { get; set; }
        public List<double> LeftElbow { get; set; }
        public List<double> RightElbow { get; set; }
        public List<double> LeftWrist { get; set; }
        public List<double> RightWrist { get; set; }
        public List<double> LeftHip { get; set; }
        public List<double> RightHip { get; set; }
        public List<double> LeftKnee { get; set; }
        public List<double> RightKnee { get; set; }
        public List<double> LeftAnkle { get; set; }
        public List<double> RightAnkle { get; set; }
    }

    public class CommandRoot
    {
        public string userid { get; set; }
        public Pose pose { get; set; }
        public List<Hand> hands { get; set; }
    }

    public class UnityPositionPose
    {
        public Vector3 Head { get; set; }
        public Vector3 Neck { get; set; }
        public Vector3 Pelvis { get; set; }//
        public Vector3 LeftClavicle { get; set; }//
        public Vector3 RightClavicle { get; set; }//
        public Vector3 LeftElbow { get; set; }//
        public Vector3 RightElbow { get; set; }//
        public Vector3 LeftWrist { get; set; }//
        public Vector3 RightWrist { get; set; }//
        public Vector3 LeftHip { get; set; }//
        public Vector3 RightHip { get; set; }//
        public Vector3 LeftKnee { get; set; }//
        public Vector3 RightKnee { get; set; }//
        public Vector3 LeftAnkle { get; set; }//
        public Vector3 RightAnkle { get; set; }//

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

        private static Vector3 ListToVector(List<double> list) => new Vector3((float)list[0], (float)list[1], (float)list[2]);
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

        private static Quaternion GetRotation(Vector3 startBone, Vector3 endBone, bool inverse = false)
        {
            var direction = endBone - startBone;
            return Quaternion.LookRotation(Vector3.Cross(direction, Vector3.right) * (inverse ? -1 : 1), direction);
        }
    }
}