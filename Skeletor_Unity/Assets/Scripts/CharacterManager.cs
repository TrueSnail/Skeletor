using SkeletonDataModel;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class CharacterManager : MonoBehaviour
{
    public static CharacterManager Instance;

    public GameObject CharacterPrefab;
    public Transform CharacterOrigin;

    private Dictionary<string, CharacterHandler> Characters = new();

    private void Awake()
    {
        Instance = this;
    }

    public void HandleCommand(List<CommandRoot> command)
    {
        foreach (var item in command)
        {
            if (!Characters.ContainsKey(item.userid)) AddCharacter(item.userid);
            Characters[item.userid].UpdatePose(item.pose);
        }

        foreach (var character in Characters)
        {
            if (!command.Select(c => c.userid).Contains(character.Key.ToString())) 
                RemoveCharacter(character.Key);
        }
    }

    private void AddCharacter(string id)
    {
        var character = Instantiate(CharacterPrefab).GetComponent<CharacterHandler>();
        character.CharacterOrigin = CharacterOrigin;
        Characters[id] = character;
    }

    private void RemoveCharacter(string id)
    {
        Destroy(Characters[id].gameObject);
        Characters.Remove(id);
    }
}
