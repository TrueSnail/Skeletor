using SkeletonDataModel;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharacterHandler : MonoBehaviour
{
    public SkeletonDataModel.Pose PoseData;

    [Header("Settings")]
    public Transform CharacterOrigin;
    public bool PinToWorldRoot = false;
    public bool DrawDebugPose = true;
    public float Spring = 10;
    public float Damper = 1;

    [Header("TEST")]
    public Transform Hips;
    public ConfigurableJoint HipsJ;
    public Transform ArmatureTest;
    public Transform LeftLegUp;
    public Transform RightLegUp;
    public ConfigurableJoint RightLegUpJ;
    public Transform Neck;
    public ConfigurableJoint NeckJ;
    public Transform LeftArm;
    public ConfigurableJoint LeftArmJ;
    public Transform LeftForearm;
    public Transform RightArm;
    public ConfigurableJoint RightArmJ;
    public Transform RightForearm;
    public ConfigurableJoint RightForearmJ;
    public Transform LeftKnee;
    public Transform RightKnee;
    public ConfigurableJoint RightKneeJ;

    private void Start()
    {
        var origin = ConfigLoader.ConfigData.CharacterOriginPosition;
        var rotation = ConfigLoader.ConfigData.CharacterRotation;
        CharacterOrigin.position = new Vector3(origin[0], origin[1], origin[2]);
        CharacterOrigin.rotation = Quaternion.Euler(rotation[0], rotation[1], rotation[2]);
        PinToWorldRoot = ConfigLoader.ConfigData.PinCharacterToOriginPosition;
        DrawDebugPose = ConfigLoader.ConfigData.DrawDebugRig;
    }

    public void UpdatePose(SkeletonDataModel.Pose pose)
    {
        PoseData = pose;
        UpdateCharacter();
    }

    private void Update()
    {
        UpdateJointDrives();
    }

    public void UpdateCharacter()
    {
        var pose = UnityRotationPose.FromPose(PoseData);
        if (!enabled) return;

        Hips.position = PinToWorldRoot ? CharacterOrigin.position : CharacterOrigin.position + pose.Position;

        var rot = Hips.localRotation;
        Hips.localRotation = LocalRotation(Hips, pose.Rotation);
        LeftLegUp.localRotation = LocalRotation(LeftLegUp, pose.LeftUpLeg);
        //RightLegUp.localRotation = LocalRotation(RightLegUp, pose.RightUpLeg);
        RightLegUpJ.targetRotation = LocalRotation(RightLegUp, pose.RightUpLeg);
        LeftKnee.localRotation = LocalRotation(LeftKnee, pose.LeftLeg);
        //RightKnee.localRotation = LocalRotation(RightKnee, pose.RightLeg);
        RightKneeJ.targetRotation = LocalRotation(RightKnee, pose.RightLeg);
        LeftArm.localRotation = LocalRotation(LeftArm, pose.LeftArm);
        //LeftArmJ.targetRotation = LocalRotation(LeftArm, pose.LeftArm);
        //RightArm.localRotation = LocalRotation(RightArm, pose.RightArm);
        RightArmJ.targetRotation = LocalRotation(RightArm, pose.RightArm);
        LeftForearm.localRotation = LocalRotation(LeftForearm, pose.LeftForeArm);
        //RightForearm.localRotation = LocalRotation(RightForearm, pose.RightForeArm);
        RightForearmJ.targetRotation = LocalRotation(RightForearm, pose.RightForeArm);
        //Neck.localRotation = LocalRotation(Neck, pose.Neck);
        NeckJ.targetRotation = LocalRotation(Neck, pose.Neck)/* * Quaternion.Euler(-90, 0, 0)*/;
        //Hips.localRotation = LocalRotation(Hips, pose.Rotation) * CharacterOrigin.rotation;
        Hips.localRotation = rot;
        HipsJ.targetRotation = LocalRotation(Hips, pose.Rotation) * CharacterOrigin.rotation;
    }

    private void UpdateJointDrives()
    {
        UpdateJointDrive(HipsJ);
        UpdateJointDrive(NeckJ);
        UpdateJointDrive(RightArmJ);
        UpdateJointDrive(RightForearmJ);
        UpdateJointDrive(RightLegUpJ);
    }

    private void UpdateJointDrive(ConfigurableJoint joint)
    {
        var drive = joint.slerpDrive;
        drive.positionSpring = Spring;
        drive.positionDamper = Damper;
        joint.slerpDrive = drive;
    }

    private void OnDrawGizmos()
    {
        if (PoseData == null || !DrawDebugPose) return;
        var pose = UnityPositionPose.FromPose(PoseData);

        float pointRadius = 0.05f;

        Gizmos.color = Color.white;
        Gizmos.DrawSphere(pose.Pelvis, pointRadius);
        Gizmos.DrawSphere(pose.LeftClavicle, pointRadius);
        Gizmos.DrawSphere(pose.RightClavicle, pointRadius);
        Gizmos.DrawSphere(pose.Neck, pointRadius);
        Gizmos.DrawSphere(pose.Head, pointRadius);

        Gizmos.DrawLine(pose.LeftHip, pose.Pelvis);
        Gizmos.DrawLine(pose.Pelvis, pose.RightHip);
        Gizmos.DrawLine(pose.RightClavicle, pose.LeftClavicle);
        Gizmos.DrawLine(pose.Neck, pose.Head);

        Gizmos.color = Color.blue;
        Gizmos.DrawSphere(pose.LeftHip, pointRadius);
        Gizmos.DrawSphere(pose.RightHip, pointRadius);
        Gizmos.DrawSphere(pose.RightKnee, pointRadius);
        Gizmos.DrawSphere(pose.LeftKnee, pointRadius);
        Gizmos.DrawSphere(pose.RightAnkle, pointRadius);
        Gizmos.DrawSphere(pose.LeftAnkle, pointRadius);

        Gizmos.DrawLine(pose.RightHip, pose.RightKnee);
        Gizmos.DrawLine(pose.LeftHip, pose.LeftKnee);
        Gizmos.DrawLine(pose.RightAnkle, pose.RightKnee);
        Gizmos.DrawLine(pose.LeftAnkle, pose.LeftKnee);

        Gizmos.color = Color.red;
        Gizmos.DrawSphere(pose.RightElbow, pointRadius);
        Gizmos.DrawSphere(pose.LeftElbow, pointRadius);
        Gizmos.DrawSphere(pose.RightWrist, pointRadius);
        Gizmos.DrawSphere(pose.LeftWrist, pointRadius);

        Gizmos.DrawLine(pose.RightElbow, pose.RightClavicle);
        Gizmos.DrawLine(pose.RightElbow, pose.RightWrist);
        Gizmos.DrawLine(pose.LeftElbow, pose.LeftClavicle);
        Gizmos.DrawLine(pose.LeftElbow, pose.LeftWrist);

    }

    private Quaternion LocalRotation(Transform transform, Quaternion worldRotation)
    {
        var parentRotation = transform.parent.rotation;
        //if (transform.parent == ArmatureTest) parentRotation = transform.parent.parent.GetComponent<ConfigurableJoint>().targetRotation;
        //if (transform.parent.TryGetComponent<ConfigurableJoint>(out var joint))
        //{
        //    parentRotation = joint.targetRotation;
        //}
        return Quaternion.Inverse(parentRotation) * worldRotation;
    }
}
