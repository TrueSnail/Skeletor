using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConfigDataModel
{
    public int UdpPort { get; private set; }
    public string[] IpAdressWhitelist { get; private set; }
    public float[] CharacterOriginPosition { get; private set; }
    public float[] CharacterRotation { get; private set; }
    public bool PinCharacterToOriginPosition { get; private set; }
    public float CharacterInterpolationSmoothness { get; private set; }
    public float CharacterSpawnDespawnParticlesCount { get; private set; }
    public bool DrawDebugRig { get; private set; }
    public bool LogReceivedUdpPackets { get; private set; }
}
