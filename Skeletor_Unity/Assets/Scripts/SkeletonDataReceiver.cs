using Newtonsoft.Json;
using SkeletonDataModel;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class SkeletonDataReceiver : MonoBehaviour
{
    public Transform CharacterOrigin;
    public bool PinToWorldRoot = false;
    public bool DrawDebugPose = true;

    [Header("TEST")]
    public Transform Hips;
    public Transform LeftLegUp;
    public Transform RightLegUp;
    public Transform LeftShoulder;
    public Transform RightShoulder;
    public Transform Neck;
    public Transform Head;
    public Transform LeftArm;
    public Transform LeftForearm;
    public Transform LeftHand;
    public Transform RightArm;
    public Transform RightForearm;
    public Transform RightHand;
    public Transform LeftKnee;
    public Transform LeftFoot;
    public Transform RightKnee;
    public Transform RightFoot;

    private UdpClient UdpClient;
    private Queue<Action> ExecuteOnMainThread = new();
    private SkeletonDataModel.Root Skeleton;

    private void Awake()
    {
        UdpClient = new(ConfigLoader.ConfigData.UdpPort);
        UdpClient.BeginReceive(OnPacketRecived, null);

        var origin = ConfigLoader.ConfigData.CharacterOriginPosition;
        var rotation = ConfigLoader.ConfigData.CharacterRotation;
        CharacterOrigin.position = new Vector3(origin[0], origin[1], origin[2]);
        CharacterOrigin.rotation = Quaternion.Euler(rotation[0], rotation[1], rotation[2]);
        PinToWorldRoot = ConfigLoader.ConfigData.PinCharacterToOriginPosition;
        DrawDebugPose = ConfigLoader.ConfigData.DrawDebugRig;
    }

    private void Update()
    {
        lock (ExecuteOnMainThread)
        {
            while (ExecuteOnMainThread.Count > 0) ExecuteOnMainThread.Dequeue().Invoke();
        }
    }

    private void OnPacketRecived(IAsyncResult result)
    {
        IPEndPoint endPoint = null;
        try
        {
            string message = Encoding.UTF8.GetString(UdpClient.EndReceive(result, ref endPoint));
            if (ConfigLoader.ConfigData.LogReceivedUdpPackets)
            {
                lock (ExecuteOnMainThread) ExecuteOnMainThread.Enqueue(() => Debug.Log($"Received UDP message: {message}, {endPoint.Address}"));
            }

            if (ConfigLoader.ConfigData.IpAdressWhitelist.Contains(endPoint.Address.ToString()))
            {
                List<SkeletonDataModel.Root> command = DeserializeMessage(message);
                lock (ExecuteOnMainThread) ExecuteOnMainThread.Enqueue(() => HandleCommand(command));
            }
        }
        catch (Exception e)
        {
            lock (ExecuteOnMainThread) ExecuteOnMainThread.Enqueue(() => Debug.LogError(e.Message));
        }

        if (UdpClient == null) return;
        UdpClient.BeginReceive(OnPacketRecived, null);
    }

    List<SkeletonDataModel.Root> DeserializeMessage(string message)
    {
        JsonSerializerSettings settings = new() { MissingMemberHandling = MissingMemberHandling.Error };
        return JsonConvert.DeserializeObject<List<SkeletonDataModel.Root>>(message, settings);
    }

    private void HandleCommand(List<SkeletonDataModel.Root> command)
    {
        Debug.Log($"Received data succesfully: {command[0].ToString()}");
        Skeleton = command[0];

        UnityRotationPose pose = UnityRotationPose.FromPose(command[0].pose);

        Hips.position = PinToWorldRoot ? CharacterOrigin.position : CharacterOrigin.position + pose.Position;

        Hips.localRotation = LocalRotation(Hips, pose.Rotation);
        LeftLegUp.localRotation = LocalRotation(LeftLegUp, pose.LeftUpLeg);
        RightLegUp.localRotation = LocalRotation(RightLegUp, pose.RightUpLeg);
        LeftKnee.localRotation = LocalRotation(LeftKnee, pose.LeftLeg);
        RightKnee.localRotation = LocalRotation(RightKnee, pose.RightLeg);
        LeftShoulder.localRotation = LocalRotation(LeftShoulder, pose.LeftArm);
        RightShoulder.localRotation = LocalRotation(RightShoulder, pose.RightArm);
        LeftForearm.localRotation = LocalRotation(LeftForearm, pose.LeftForeArm);
        RightForearm.localRotation = LocalRotation(RightForearm, pose.RightForeArm);
        Neck.localRotation = LocalRotation(Neck, pose.Neck);
        Hips.localRotation = LocalRotation(Hips, pose.Rotation) * CharacterOrigin.rotation;

        //Hips.position = PinToWorldRoot ? CharacterOrigin.position : CharacterOrigin.position + pose.Position;
        //Hips.rotation = pose.Rotation;
        //LeftLegUp.rotation = pose.LeftUpLeg;
        //RightLegUp.rotation = pose.RightUpLeg;
        //LeftKnee.rotation = pose.LeftLeg;
        //RightKnee.rotation = pose.RightLeg;
        //LeftShoulder.rotation = pose.LeftArm;
        //RightShoulder.rotation = pose.RightArm;
        //LeftForearm.rotation = pose.LeftForeArm;
        //RightForearm.rotation = pose.RightForeArm;
        //Neck.rotation = pose.Neck;

        //TEMP
        //Hips.transform.position = TEMPCONVERT(command[0].pose.Pelvis);
        //LeftLegUp.transform.position = TEMPCONVERT(command[0].pose.LeftHip);
        //RightLegUp.transform.position = TEMPCONVERT(command[0].pose.RightHip);
        //LeftShoulder.transform.position = TEMPCONVERT(command[0].pose.LeftClavicle);
        //RightShoulder.transform.position = TEMPCONVERT(command[0].pose.RightClavicle);
        //Neck.transform.position = TEMPCONVERT(command[0].pose.Neck);
        //Head.transform.position = TEMPCONVERT(command[0].pose.Head);
        //LeftArm.transform.position = TEMPCONVERT(command[0].pose.LeftElbow);
        ////LeftForearm.transform.position = TEMPCONVERT(command[0].pose.left
        //LeftHand.transform.position = TEMPCONVERT(command[0].pose.LeftWrist);
        //RightArm.transform.position = TEMPCONVERT(command[0].pose.RightElbow);
        ////RightForearm.transform.position = TEMPCONVERT(command[0].pose
        //RightHand.transform.position = TEMPCONVERT(command[0].pose.RightWrist);
        //LeftKnee.transform.position = TEMPCONVERT(command[0].pose.LeftKnee);
        //LeftFoot.transform.position = TEMPCONVERT(command[0].pose.LeftAnkle);
        //RightKnee.transform.position = TEMPCONVERT(command[0].pose.RightKnee);
        //RightFoot.transform.position = TEMPCONVERT(command[0].pose.RightAnkle);
    }

    private Quaternion LocalRotation(Transform transform, Quaternion worldRotation) => Quaternion.Inverse(transform.parent.rotation) * worldRotation;

    private void OnDestroy()
    {
        if (UdpClient != null)
        {
            UdpClient.Close();
            UdpClient.Dispose();
        }
    }

    private void OnDrawGizmos()
    {
        if (Skeleton == null || !DrawDebugPose) return;
        var pose = UnityPositionPose.FromPose(Skeleton.pose);

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
}
