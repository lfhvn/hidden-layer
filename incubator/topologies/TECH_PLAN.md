# Technical Plan — Latent Topologies

## Architecture Overview
```
[Corpus] → [Local Embedding Model] → [SQLite + ANN]
      ↘                             ↗
       [UMAP coords → Three.js Scene → UI + Audio/Haptics]
```

### Stack
- **Frontend:** React Native + react-three-fiber + expo-av
- **Model:** MiniLM / E5-small → Core ML / TFLite int8
- **Storage:** SQLite (embeddings, coords, graph, notes)
- **ANN:** SIMD cosine or HNSW native module
- **Audio:** Tone.js or AVAudioEngine
- **Haptics:** expo-haptics

### Data Schema
```sql
item(id,text,meta)
embedding(item_id,vec BLOB)
coord2d(item_id,x,y)
edge(src,dst,weight)
note(id,item_id,body)
```

### Projection Placement
Use **IDW interpolation** or lightweight **param-UMAP MLP** for new points.

### Performance Targets
- Cold start < 5 s
- Embed latency < 50 ms
- 60 fps render (≤ 10 k points)

### Security & Privacy
All inference on-device, no network I/O, model card visible.

### Future Extensions
- Collaborative maps
- Temporal embedding tracking
- Biofeedback inputs
