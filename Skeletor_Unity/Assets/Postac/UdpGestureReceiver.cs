using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class UdpGestureReceiver : MonoBehaviour
{
    [Header("UDP Settings")]
    public int listenPort = 5005;

    [Header("Animator")]
    public Animator animator;
    public string gestureParameterName = "Gesture";

    [Header("Gestures")]
    [Tooltip("Czas trwania gestu zanim postaæ wróci do Idle")]
    public float gestureDuration = 1.0f;  // dopasuj do d³ugoœci animacji

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private string lastGesture = "";

    private Coroutine revertCoroutine;

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();

        remoteEndPoint = new IPEndPoint(IPAddress.Any, listenPort);
        udpClient = new UdpClient(listenPort);
        udpClient.Client.Blocking = false;

        Debug.Log($"UDP listener started on port {listenPort}");

        // Na start wymuœ Idle
        SetGesture(0);
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

                    PlayGestureOnce(code);
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

    private void PlayGestureOnce(int gestureCode)
    {
        // Idle (0) – po prostu ustawiamy i nic wiêcej
        if (gestureCode == 0)
        {
            // zatrzymaj ewentualny timer powrotu
            if (revertCoroutine != null)
            {
                StopCoroutine(revertCoroutine);
                revertCoroutine = null;
            }

            SetGesture(0);
            return;
        }

        // Ustawiamy gest
        SetGesture(gestureCode);

        // Restartujemy timer powrotu do Idle
        if (revertCoroutine != null)
            StopCoroutine(revertCoroutine);

        revertCoroutine = StartCoroutine(RevertToIdleAfterDelay());
    }

    private void SetGesture(int code)
    {
        if (animator != null)
        {
            animator.SetInteger(gestureParameterName, code);
        }
    }

    private IEnumerator RevertToIdleAfterDelay()
    {
        yield return new WaitForSeconds(gestureDuration);
        SetGesture(0);  // powrót do Idle
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
