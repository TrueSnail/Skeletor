using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CharacterParticles : MonoBehaviour
{
    public GameObject ParticlesPrefab;
    public SkinnedMeshRenderer MeshRenderer;

    public float ParticlesCount;

    private void Start()
    {
        ParticlesCount = ConfigLoader.ConfigData.CharacterSpawnDespawnParticlesCount;
        Invoke("SpawnParticles", 0.1f);
    }

    private void OnDestroy()
    {
        if (!gameObject.scene.isLoaded) return;
        SpawnParticles();
    }

    public void SpawnParticles()
    {
        var particles = Instantiate(ParticlesPrefab, transform.position, Quaternion.identity);
        var system = particles.GetComponent<ParticleSystem>();
        var emission = system.emission;
        var shape = system.shape;
        shape.skinnedMeshRenderer = MeshRenderer;
        emission.rateOverTime = new ParticleSystem.MinMaxCurve(ParticlesCount);
        system.Play();
    }
}
