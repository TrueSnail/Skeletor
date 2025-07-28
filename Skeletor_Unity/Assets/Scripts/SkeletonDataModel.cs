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

    public class Root
    {
        public string userid { get; set; }
        public Pose pose { get; set; }
        public List<Hand> hands { get; set; }
    }
}