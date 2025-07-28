using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Skeletor_PacketRecorderPlayer;

public class ConfigDataModel
{
    public int UdpPort { get; private set; }
    public string[]? RecordingPlaylist { get; private set; }
    public bool IsLoggingEnabled { get; private set; }
}
