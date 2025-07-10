using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConfigDataModel
{
    public int UdpPort { get; private set; }
    public string[] IpAdressWhitelist { get; private set; }
    public bool LogReceivedUdpPackets { get; private set; }
}
