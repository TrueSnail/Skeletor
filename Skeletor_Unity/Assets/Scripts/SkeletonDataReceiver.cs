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
    public bool PinToWorldRoot = false;

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

    private void Awake()
    {
        UdpClient = new(ConfigLoader.ConfigData.UdpPort);
        UdpClient.BeginReceive(OnPacketRecived, null);
    }

    private void FixedUpdate()
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

        UnityRotationPose pose = UnityRotationPose.FromPose(command[0].pose);

        Hips.position = PinToWorldRoot ? Vector3.zero : pose.Position;
        LeftLegUp.rotation = pose.LeftUpLeg;
        RightLegUp.rotation = pose.RightUpLeg;
        LeftKnee.rotation = pose.LeftLeg;
        RightKnee.rotation = pose.RightLeg;
        LeftShoulder.rotation = pose.LeftArm;
        RightShoulder.rotation = pose.RightArm;
        LeftForearm.rotation = pose.LeftForeArm;
        RightForearm.rotation = pose.RightForeArm;
        Neck.rotation = pose.Neck;

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

    private Vector3 TEMPCONVERT(List<double> list)
    {
        return new Vector3((float)list[0], (float)list[1], (float)list[2]);
    }

    private void OnDestroy()
    {
        if (UdpClient != null)
        {
            UdpClient.Close();
            UdpClient.Dispose();
        }
    }

}
