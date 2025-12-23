using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharacterParticles : MonoBehaviour
{
    public GameObject ParticlesPrefab;
    public Transform SpawnTransform;

    private void Start()
    {
        Invoke("SpawnParticles", 0.1f);
    }

    private void OnDestroy()
    {
        if (!gameObject.scene.isLoaded) return;
        SpawnParticles();
    }

    public void SpawnParticles()
    {
        var particles = Instantiate(ParticlesPrefab, SpawnTransform.position, Quaternion.identity);
    }
}
