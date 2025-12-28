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

    [Header("TEST")]
    public Transform Hips;
    public Transform LeftLegUp;
    public Transform RightLegUp;
    public Transform Neck;
    public Transform LeftArm;
    public Transform LeftForearm;
    public Transform RightArm;
    public Transform RightForearm;
    public Transform LeftKnee;
    public Transform RightKnee;

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

    public void UpdateCharacter()
    {
        var pose = UnityRotationPose.FromPose(PoseData);
        if (!enabled) return;

        Hips.position = PinToWorldRoot ? CharacterOrigin.position : CharacterOrigin.position + pose.Position;

        Hips.localRotation = LocalRotation(Hips, pose.Rotation);
        LeftLegUp.localRotation = LocalRotation(LeftLegUp, pose.LeftUpLeg);
        RightLegUp.localRotation = LocalRotation(RightLegUp, pose.RightUpLeg);
        LeftKnee.localRotation = LocalRotation(LeftKnee, pose.LeftLeg);
        RightKnee.localRotation = LocalRotation(RightKnee, pose.RightLeg);
        LeftArm.localRotation = LocalRotation(LeftArm, pose.LeftArm);
        RightArm.localRotation = LocalRotation(RightArm, pose.RightArm);
        LeftForearm.localRotation = LocalRotation(LeftForearm, pose.LeftForeArm);
        RightForearm.localRotation = LocalRotation(RightForearm, pose.RightForeArm);
        Neck.localRotation = LocalRotation(Neck, pose.Neck);
        Hips.localRotation = LocalRotation(Hips, pose.Rotation) * CharacterOrigin.rotation;
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

    private Quaternion LocalRotation(Transform transform, Quaternion worldRotation) => Quaternion.Inverse(transform.parent.rotation) * worldRotation;
}
