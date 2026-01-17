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
    public float SmoothTime;

    [Header("References")]
    public RotationInterpolator IHips;
    public RotationInterpolator ILeftLegUp;
    public RotationInterpolator IRightLegUp;
    public RotationInterpolator INeck;
    public RotationInterpolator ILeftArm;
    public RotationInterpolator ILeftForearm;
    public RotationInterpolator IRightArm;
    public RotationInterpolator IRightForearm;
    public RotationInterpolator ILeftKnee;
    public RotationInterpolator IRightKnee;

    private Vector3 CurrentPositionVelocityRef;

    private void Start()
    {
        var origin = ConfigLoader.ConfigData.CharacterOriginPosition;
        var rotation = ConfigLoader.ConfigData.CharacterRotation;
        CharacterOrigin.position = new Vector3(origin[0], origin[1], origin[2]);
        CharacterOrigin.rotation = Quaternion.Euler(rotation[0], rotation[1], rotation[2]);
        PinToWorldRoot = ConfigLoader.ConfigData.PinCharacterToOriginPosition;
        DrawDebugPose = ConfigLoader.ConfigData.DrawDebugRig;
        SmoothTime = ConfigLoader.ConfigData.CharacterInterpolationSmoothness;
    }

    private void Update()
    {
        UpdatePosition();
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

        var hipsRotation = IHips.transform.localRotation;
        IHips.transform.localRotation = LocalRotation(IHips.transform, pose.Rotation);

        ILeftLegUp.SetTargetRotation(LocalRotation(ILeftLegUp.transform, pose.LeftUpLeg));
        IRightLegUp.SetTargetRotation(LocalRotation(IRightLegUp.transform, pose.RightUpLeg));
        ILeftKnee.SetTargetRotation(LocalRotation(ILeftKnee.transform, pose.LeftLeg));
        IRightKnee.SetTargetRotation(LocalRotation(IRightKnee.transform, pose.RightLeg));
        ILeftArm.SetTargetRotation(LocalRotation(ILeftArm.transform, pose.LeftArm));
        IRightArm.SetTargetRotation(LocalRotation(IRightArm.transform, pose.RightArm));
        ILeftForearm.SetTargetRotation(LocalRotation(ILeftForearm.transform, pose.LeftForeArm));
        IRightForearm.SetTargetRotation(LocalRotation(IRightForearm.transform, pose.RightForeArm));
        INeck.SetTargetRotation(LocalRotation(INeck.transform, pose.Neck));

        IHips.transform.localRotation = hipsRotation;
        IHips.SetTargetRotation(LocalRotation(IHips.transform, pose.Rotation) * CharacterOrigin.rotation);
    }

    private void UpdatePosition()
    {
        if (!enabled) return;
        var pose = UnityRotationPose.FromPose(PoseData);
        var targetPosition = PinToWorldRoot ? CharacterOrigin.position : CharacterOrigin.position + pose.Position;
        IHips.transform.position = Vector3.SmoothDamp(IHips.transform.position, targetPosition, ref CurrentPositionVelocityRef, SmoothTime);

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
