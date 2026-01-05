using Skeletor_PacketRecorderPlayer;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Tomlyn;

ConfigDataModel config = GetConfig();
UdpClient client = new UdpClient(2424);

while (true)
{
    foreach (var recordingName in config.RecordingPlaylist!)
    {
        var recording = GetRecording(recordingName)!;
        int i = 0;
        foreach (var dataPacket in recording)
        {
            i++;
            SendText(dataPacket.Data);
            if (config.IsLoggingEnabled) Console.WriteLine($"Sending packet from {recordingName} numbered {i} ({MathF.Round((float)i / recording.Length * 100, 2)}%)");
            await Task.Delay(dataPacket.DelayMs);
        }
        if (config.IsLoggingEnabled) Console.WriteLine("");
    }
}

void SendText(string text)
{
    var encodedText = Encoding.UTF8.GetBytes(text);
    client.Send(encodedText, "127.0.0.1", config.UdpPort);
}

RecordingDataModel[]? GetRecording(string recordingName)
{
    string text = File.ReadAllText(Environment.CurrentDirectory + @"\Recordings\" + recordingName + ".json");
    return JsonSerializer.Deserialize<RecordingDataModel[]>(text);
}

ConfigDataModel GetConfig()
{
    string text = File.ReadAllText(Environment.CurrentDirectory + @"\Config.toml");
    var options = new TomlModelOptions() { ConvertPropertyName = s => s };
    return Toml.ToModel<ConfigDataModel>(text, options: options);
}