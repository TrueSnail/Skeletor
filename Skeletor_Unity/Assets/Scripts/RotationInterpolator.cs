using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotationInterpolator : MonoBehaviour
{
    private Quaternion TargetRotation;
    private Quaternion DerivSmoothDampRef;

    public float SmoothTime;

    private void Start()
    {
        SmoothTime = ConfigLoader.ConfigData.CharacterInterpolationSmoothness;
    }

    public void SetTargetRotation(Quaternion localRotatation)
    {
        TargetRotation = localRotatation;
    }

    private void Update()
    {
        transform.localRotation = QuaternionUtil.SmoothDamp(transform.localRotation, TargetRotation, ref DerivSmoothDampRef, SmoothTime);
    }
}
