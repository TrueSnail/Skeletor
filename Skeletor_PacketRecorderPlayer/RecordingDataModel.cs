using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Skeletor_PacketRecorderPlayer;

public class RecordingDataModel
{
    public required string Data { get; init; }
    public int DelayMs { get; init; }
}
