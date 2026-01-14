using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class NPCGestureController : MonoBehaviour
{
    [Header("Animator")]
    public Animator animator;
    public string gestureParameterName = "Gesture";

    [Header("Gestures")]
    [Tooltip("Czas trwania gestu zanim postaæ wróci do Idle")]
    public float gestureDuration = 1.0f;  // dopasuj do d³ugoœci animacji
    private Coroutine revertCoroutine;

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();

        // Na start wymuœ Idle
        SetGesture(0);
    }
        //Invoked by unity event from UdpGestureReciver
    public void PlayGestureOnce(int gestureCode)
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

}
