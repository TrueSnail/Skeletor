using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.Events;

public class UdpGestureReceiver : MonoBehaviour
{
    [Header("UDP Settings")]
    public int listenPort = 5005;

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;

    public UnityEvent<int> RecivedGestureCode;

    void Start()
    {

        remoteEndPoint = new IPEndPoint(IPAddress.Any, listenPort);
        udpClient = new UdpClient(listenPort);
        udpClient.Client.Blocking = false;

        Debug.Log($"UDP listener started on port {listenPort}");
        
    }

    void Update()
    {
        ReceiveAllPendingPackets();
    }

    private void ReceiveAllPendingPackets()
    {
        try
        {
            while (udpClient.Available > 0)
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string json = Encoding.UTF8.GetString(data);

                string gesture = ParseGestureFromJson(json);
                if (!string.IsNullOrEmpty(gesture))
                {
                    gesture = gesture.ToLower();                  
                    int code = MapGestureToInt(gesture);

                    RecivedGestureCode.Invoke(code);
                }
            }
        }
        catch (SocketException ex)
        {
            Debug.LogWarning("UDP socket exception: " + ex.Message);
        }
    }

    private string ParseGestureFromJson(string json)
    {
        const string key = "\"gesture\"";
        int keyIndex = json.IndexOf(key);
        if (keyIndex == -1) return null;

        int colonIndex = json.IndexOf(':', keyIndex);
        if (colonIndex == -1) return null;

        int firstQuote = json.IndexOf('"', colonIndex + 1);
        int secondQuote = json.IndexOf('"', firstQuote + 1);

        if (firstQuote == -1 || secondQuote == -1) return null;

        string value = json.Substring(firstQuote + 1, secondQuote - firstQuote - 1);
        return value;
    }

    private int MapGestureToInt(string gesture)
    {
        switch (gesture)
        {
            case "wave":
                return 1;
            case "thumbs_up":
                return 2;
            case "ok":
                return 3;
            case "arms_up":
                return 4;
            case "idle":
            case "":
            default:
                return 0;
        }
    }

    private void OnApplicationQuit()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }
    }
}
