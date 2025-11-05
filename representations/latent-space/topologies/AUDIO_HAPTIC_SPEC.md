# Audio & Haptic Mapping Specification

**Precise mathematical definitions** for translating latent space geometry into auditory and haptic feedback.

**Design Principle**: Each mapping must be **deterministic, testable, and perceptually meaningful**.

---

## Core Concepts

### Semantic Properties (Input)
From embedding space and UMAP projection:

```python
# Per-item properties
position: (x, y)           # 2D UMAP coordinates
embedding: vec[384]        # High-D semantic vector
density: float             # Local point density (kNN-based)
cluster_id: int            # Cluster assignment (-1 = noise)

# Relational properties (between items A and B)
distance_2d: float         # Euclidean distance in UMAP space
similarity: float          # Cosine similarity in embedding space
interpolation_t: float     # Position along A→B path (0 to 1)
```

### Perceptual Properties (Output)
Audio and haptic parameters:

```python
# Audio
pitch: float               # Frequency in Hz
volume: float              # Amplitude 0-1
timbre: str                # Waveform type or sample
duration: int              # Milliseconds

# Haptic
intensity: float           # Vibration strength 0-1
pattern: str               # Vibration pattern ID
duration: int              # Milliseconds
```

---

## 1. Audio Mappings

### 1.1 Pitch from Semantic Similarity

**Rationale**: Similar concepts = consonant intervals; dissimilar = dissonant.

**Formula**:
```python
def pitch_from_similarity(sim: float, base_hz: float = 440.0) -> float:
    """
    Map cosine similarity to musical interval.

    sim = 1.0 → unison (ratio 1:1)
    sim = 0.9 → major third (ratio 5:4)
    sim = 0.7 → perfect fifth (ratio 3:2)
    sim = 0.5 → octave (ratio 2:1)
    sim < 0.5 → dissonant intervals
    """
    if sim >= 0.9:
        ratio = 1.0 + (1.0 - sim) * 0.25  # 1.0 → 1.025 (near unison)
    elif sim >= 0.7:
        ratio = 1.25  # Major third
    elif sim >= 0.5:
        ratio = 1.5   # Perfect fifth
    else:
        ratio = 2.0 + (0.5 - sim) * 2.0  # Octave and beyond

    return base_hz * ratio
```

**Test**: Items with sim=1.0 should sound identical; sim=0.0 should sound maximally different.

---

### 1.2 Pitch from 2D Distance

**Rationale**: Proximity in visual space = proximity in pitch space.

**Formula**:
```python
def pitch_from_distance(dist: float,
                       min_hz: float = 220.0,
                       max_hz: float = 880.0,
                       max_dist: float = 10.0) -> float:
    """
    Map 2D distance to pitch (logarithmic scale).

    dist = 0 → max_hz (closest = highest pitch)
    dist = max_dist → min_hz (farthest = lowest pitch)
    """
    import math

    # Normalize distance to [0, 1]
    t = min(dist / max_dist, 1.0)

    # Logarithmic mapping (perceptually linear)
    log_min = math.log2(min_hz)
    log_max = math.log2(max_hz)

    # Invert so closer = higher pitch
    log_hz = log_max - t * (log_max - log_min)

    return 2 ** log_hz
```

**Test**: Moving from point A to nearby point B should produce smooth pitch glide.

---

### 1.3 Interpolation Scrubber (Path Sonification)

**Rationale**: Scrubbing between two concepts = continuous pitch glide.

**Formula**:
```python
def interpolate_pitch(pitch_a: float, pitch_b: float, t: float) -> float:
    """
    Smooth pitch transition using logarithmic interpolation.

    t = 0.0 → pitch_a
    t = 1.0 → pitch_b
    """
    import math

    log_a = math.log2(pitch_a)
    log_b = math.log2(pitch_b)
    log_interp = log_a + t * (log_b - log_a)

    return 2 ** log_interp

def interpolate_text(emb_a: np.ndarray, emb_b: np.ndarray,
                    t: float, corpus_embeddings: np.ndarray) -> str:
    """
    Find nearest corpus item to interpolated embedding.
    """
    # Linear interpolation in embedding space
    emb_interp = (1 - t) * emb_a + t * emb_b

    # L2 normalize
    emb_interp = emb_interp / np.linalg.norm(emb_interp)

    # Find nearest neighbor
    sims = cosine_similarity(emb_interp.reshape(1, -1), corpus_embeddings)[0]
    idx = np.argmax(sims)

    return corpus_items[idx].text
```

**Implementation** (React Native):
```typescript
// Scrubber component
const [scrubPosition, setScrubPosition] = useState(0.0); // 0 to 1

useEffect(() => {
  const hz = interpolatePitch(pitchA, pitchB, scrubPosition);
  playTone(hz, duration=100); // Continuous tone

  const interpolatedText = findInterpolatedText(embA, embB, scrubPosition);
  setDisplayText(interpolatedText);
}, [scrubPosition]);
```

---

### 1.4 Timbre from Cluster Density

**Rationale**: Dense clusters = smooth tones; sparse regions = harsh/noisy tones.

**Formula**:
```python
def timbre_from_density(density: float,
                       min_density: float = 0.1,
                       max_density: float = 5.0) -> str:
    """
    Map local density to waveform type.

    High density → sine wave (smooth)
    Mid density → triangle wave
    Low density → sawtooth wave (harsh)
    """
    t = (density - min_density) / (max_density - min_density)
    t = np.clip(t, 0.0, 1.0)

    if t > 0.66:
        return "sine"
    elif t > 0.33:
        return "triangle"
    else:
        return "sawtooth"
```

**Alternative**: Use additive synthesis with harmonics proportional to density.

---

### 1.5 Volume from Attention/Selection

**Rationale**: Selected items louder; background quieter.

**Formula**:
```python
def volume_from_state(is_selected: bool,
                     is_neighbor: bool,
                     base_volume: float = 0.3) -> float:
    """
    Adaptive volume based on selection state.
    """
    if is_selected:
        return 1.0
    elif is_neighbor:
        return 0.6
    else:
        return base_volume
```

---

## 2. Haptic Mappings

### 2.1 Boundary Crossing Detection

**Rationale**: Haptic feedback marks transition between semantic regions.

**Formula**:
```python
def detect_boundary_crossing(position_t0: Tuple[float, float],
                            position_t1: Tuple[float, float],
                            cluster_map: np.ndarray,
                            resolution: float = 0.1) -> bool:
    """
    Check if movement crossed cluster boundary.

    Returns: True if cluster_id changed along path.
    """
    x0, y0 = position_t0
    x1, y1 = position_t1

    # Sample points along path
    steps = int(np.hypot(x1 - x0, y1 - y0) / resolution) + 1

    cluster_prev = get_cluster_at(x0, y0, cluster_map)

    for i in range(1, steps):
        t = i / steps
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        cluster_curr = get_cluster_at(x, y, cluster_map)

        if cluster_curr != cluster_prev and cluster_curr != -1:
            return True

    return False
```

**Haptic Pattern**:
```typescript
// On boundary crossing
if (detectBoundaryCrossing(prevPos, currPos, clusterMap)) {
  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
}
```

---

### 2.2 Density Gradient → Vibration Intensity

**Rationale**: Approaching cluster core = increasing vibration.

**Formula**:
```python
def haptic_intensity_from_gradient(density: float,
                                   density_gradient: float,
                                   threshold: float = 0.5) -> float:
    """
    Vibration intensity from density gradient magnitude.

    Large gradient (approaching cluster) → strong vibration
    Small gradient (within cluster) → gentle vibration
    """
    gradient_magnitude = abs(density_gradient)

    # Normalize to [0, 1]
    intensity = np.clip(gradient_magnitude / threshold, 0.0, 1.0)

    return intensity
```

**Implementation**:
```typescript
const gradient = computeDensityGradient(position, densityMap);
const intensity = hapticIntensityFromGradient(density, gradient);

if (intensity > 0.1) {
  Haptics.selectionAsync(); // Light tap
} else if (intensity > 0.5) {
  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
}
```

---

### 2.3 Long-Press → Neighbor Vibration Pattern

**Rationale**: Rhythmic feedback indicates number of neighbors.

**Formula**:
```python
def neighbor_vibration_pattern(num_neighbors: int, max_neighbors: int = 12) -> List[int]:
    """
    Generate vibration pattern encoding neighbor count.

    Returns: List of [vibrate_ms, pause_ms, ...] intervals
    """
    # Short pulse for each neighbor (max 5 pulses)
    num_pulses = min(num_neighbors // 3, 5)

    pattern = []
    for i in range(num_pulses):
        pattern.extend([50, 100])  # 50ms vibrate, 100ms pause

    return pattern
```

---

### 2.4 Selection Confirmation

**Rationale**: Distinct haptic feedback confirms item selection.

**Pattern**:
```typescript
// On item tap
const onSelectItem = (itemId: string) => {
  // Double tap pattern
  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  setTimeout(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }, 100);

  // Then show details drawer
  showDetailsDrawer(itemId);
};
```

---

## 3. Combined Multi-Modal Feedback

### 3.1 Exploration Mode (Free Navigation)

```typescript
const onPositionChange = (newPos: {x: number, y: number}) => {
  // Audio: Pitch from nearest point distance
  const nearest = findNearestPoint(newPos);
  const dist = distance(newPos, nearest.position);
  const hz = pitchFromDistance(dist);
  playTone(hz, volume=0.3);

  // Haptic: Boundary crossing
  if (detectBoundaryCrossing(prevPos, newPos)) {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }

  // Haptic: Density gradient
  const gradient = computeDensityGradient(newPos);
  if (gradient > threshold) {
    Haptics.selectionAsync();
  }
};
```

---

### 3.2 Interpolation Mode (Scrubber Active)

```typescript
const onScrubberMove = (t: number) => {
  // Audio: Smooth pitch glide
  const hz = interpolatePitch(pitchA, pitchB, t);
  playTone(hz, volume=0.8);

  // Haptic: At midpoint (t=0.5)
  if (Math.abs(t - 0.5) < 0.05) {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  }

  // Visual: Show interpolated text
  const text = findInterpolatedText(embA, embB, t);
  setInterpolatedText(text);
};
```

---

## 4. Accessibility Considerations

### 4.1 Audio Descriptions

```typescript
// VoiceOver / TalkBack support
const getItemDescription = (item: Item) => {
  return `${item.text}. Topic: ${item.topic}. Has ${item.neighbors.length} neighbors.`;
};
```

### 4.2 Adjustable Sensitivity

```typescript
// User preferences
interface AudioHapticSettings {
  audioEnabled: boolean;
  hapticEnabled: boolean;
  hapticSensitivity: number;  // 0.5 to 2.0
  volumeMultiplier: number;   // 0.0 to 1.0
  pitchRange: {min: number, max: number};
}
```

---

## 5. Validation & Testing

### 5.1 Perceptual Tests

**Consistency Test**:
- User hears two tones for same item → should sound identical
- User hears tones for neighbors → should sound similar
- User hears tones for distant items → should sound different

**Ordering Test**:
- User drags from A → B → C → D
- Pitch should change monotonically (all increasing or all decreasing)

**Boundary Test**:
- User crosses cluster boundary
- Should feel haptic feedback within 100ms

---

### 5.2 Performance Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| Audio latency | <50ms | Time from position change to tone |
| Haptic latency | <30ms | Time from boundary cross to vibration |
| Pitch accuracy | ±5 Hz | Deviation from formula |
| Frame rate (with audio) | >50 fps | Visual smoothness |

---

## 6. Implementation Libraries

### React Native

```bash
npm install expo-av expo-haptics
```

```typescript
import { Audio } from 'expo-av';
import * as Haptics from 'expo-haptics';

// Initialize audio
const sound = new Audio.Sound();
await sound.loadAsync(require('./assets/sine-wave.wav'));

// Play tone
await sound.setPositionAsync(0);
await sound.setVolumeAsync(volume);
await sound.playAsync();

// Trigger haptic
await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
```

---

## 7. Future Enhancements

- **Spatial Audio**: Use iOS spatial audio APIs for 3D positioning
- **Adaptive Timbre**: Learn user preferences for density→timbre mapping
- **Gestural Haptics**: Different patterns for different gestures (tap, swipe, pinch)
- **Sonification Modes**: Toggle between distance-based and similarity-based pitch

---

## References

- **Auditory Display**: https://icad.org/
- **Data Sonification**: Hermann et al., "The Sonification Handbook" (2011)
- **Haptic Design**: MacLean, "Haptic Interaction Design for Everyday Interfaces" (2008)
- **Expo Audio**: https://docs.expo.dev/versions/latest/sdk/audio/
- **Expo Haptics**: https://docs.expo.dev/versions/latest/sdk/haptics/
