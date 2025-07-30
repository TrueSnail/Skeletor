using System.Collections;
using System.Collections.Generic;
using System.IO;
using Tomlyn;
using UnityEngine;

public static class ConfigLoader
{
    public static ConfigDataModel ConfigData { get; private set; }

    private const string CONFIG_PATH = "/Config.toml";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSplashScreen)]
    public static void LoadConfig()
    {
        string configText = File.ReadAllText(Application.streamingAssetsPath + CONFIG_PATH);

        var parseOptions = new TomlModelOptions() { ConvertPropertyName = s => s };
        ConfigData = Toml.ToModel<ConfigDataModel>(configText, options: parseOptions);
    }
}
