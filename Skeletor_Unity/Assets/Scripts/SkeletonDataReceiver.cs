using Newtonsoft.Json;
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
                SkeletonDataModel command = DeserializeMessage(message);
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

    SkeletonDataModel DeserializeMessage(string message)
    {
        JsonSerializerSettings settings = new() { MissingMemberHandling = MissingMemberHandling.Error };
        return JsonConvert.DeserializeObject<SkeletonDataModel>(message, settings);
    }

    private void HandleCommand(SkeletonDataModel command)
    {
        Debug.Log($"Received data succesfully: {command.TEST}");
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
