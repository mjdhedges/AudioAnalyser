# Next Level Optimization Plan

**Status:** Current optimizations deliver 400-500x speedup  
**Target:** Additional 5-50x improvement for specific use cases  
**Focus:** Advanced techniques for production-scale processing

---

## Current Performance Baseline

After two rounds of optimization:
- **Processing speed:** 400-500x faster than original
- **Memory usage:** 66% reduction (~500 MB peak)
- **Filter operations:** Reduced from 1,851 to 39 per track
- **Vectorization:** Core loops optimized

---

## Next Level Optimizations (Priority Order)

### Priority 1: Parallel Octave Bank Processing 🔥

**Impact:** 4-8x speedup on multi-core systems  
**Difficulty:** Medium  
**Implementation time:** 2-3 hours

**Current State:**
- Octave filters applied sequentially (one frequency at a time)
- Total time: sum of all filter operations

**Proposed Solution:**
```python
# In OctaveBandFilter class
def create_octave_bank_parallel(self, audio_data, num_workers=None):
    """Create octave bank using parallel processing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    
    if num_workers is None:
        num_workers = min(len(self.OCTAVE_CENTER_FREQUENCIES), multiprocessing.cpu_count())
    
    filtered_signals = [audio_data]
    
    # Create worker function for parallel execution
    def filter_worker(args):
        audio, freq, sr = args
        # Duplicate filter logic here (no pickle issues with functions)
        from scipy import signal
        bandwidth = freq / np.sqrt(2)
        # ... filter design ...
        filtered = signal.filtfilt(b, a, audio)
        return (freq, filtered)
    
    # Process frequencies in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(filter_worker, (audio_data, freq, self.sample_rate)): freq
            for freq in self.OCTAVE_CENTER_FREQUENCIES
            if freq < self.sample_rate / 2
        }
        
        results = {}
        for future in as_completed(futures):
            freq, filtered = future.result()
            results[freq] = filtered
    
    # Reassemble in correct order
    for freq in self.OCTAVE_CENTER_FREQUENCIES:
        if freq < self.sample_rate / 2:
            filtered_signals.append(results[freq])
    
    return np.column_stack(filtered_signals)
```

**Benefits:**
- Utilizes all CPU cores
- 4-8x faster on 8-core system
- Transparent to rest of code
- Fallback to sequential if parallel fails

**Challenges:**
- Need to handle pickling (audio data must be serializable)
- Slightly more complex error handling
- Memory overhead for multiple processes

---

### Priority 2: Float32 Memory Optimization 💾

**Impact:** 50% memory reduction, 10-20% speed improvement  
**Difficulty:** Easy  
**Implementation time:** 30 minutes

**Current State:**
- Audio loaded as float64 by default
- ~466 MB per octave bank

**Proposed Solution:**
```python
# In AudioProcessor.load_audio()
def load_audio(self, file_path, dtype=np.float32):
    """Load audio with specified dtype."""
    audio_data, sr = librosa.load(
        str(file_path), 
        sr=self.sample_rate, 
        dtype=dtype
    )
    return audio_data, sr

# In config.toml add:
[performance]
use_float32 = true  # Use float32 instead of float64
```

**Benefits:**
- 50% memory reduction (233 MB vs 466 MB)
- Some operations faster with smaller data
- Still sufficient precision for audio analysis

**Challenges:**
- Need to verify precision is sufficient
- May need warnings if clipping detected

---

### Priority 3: Adaptive Chunking & Batch Processing 📊

**Impact:** 2-3x for batch operations, better memory management  
**Difficulty:** Medium  
**Implementation time:** 2-3 hours

**Current State:**
- Fixed 2-second chunks
- Processes one track at a time

**Proposed Solution:**
```python
# In main.py
def analyze_tracks_batch(track_paths, output_dir, max_concurrent=4):
    """Process tracks in parallel batches."""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(analyze_single_track, track, output_dir, ...): track
            for track in track_paths
        }
        
        for future in as_completed(futures):
            track = futures[future]
            try:
                success = future.result()
                logger.info(f"Completed: {track}")
            except Exception as e:
                logger.error(f"Failed {track}: {e}")

# Adaptive chunk sizing based on file duration
def get_optimal_chunk_duration(duration_seconds):
    """Return optimal chunk duration based on file length."""
    if duration_seconds < 60:
        return 2.0  # Short files: 2s chunks
    elif duration_seconds < 300:
        return 5.0  # Medium files: 5s chunks
    else:
        return 10.0  # Long files: 10s chunks (fewer chunks)
```

**Benefits:**
- Process multiple tracks simultaneously
- Adapt to file characteristics
- Better resource utilization

**Challenges:**
- Memory management with concurrent processing
- Need to handle failures gracefully
- Thread-safety considerations

---

### Priority 4: Smart Caching & Incremental Processing 🎯

**Impact:** Near-instant re-analysis, resume capability  
**Difficulty:** Medium-Hard  
**Implementation time:** 4-5 hours

**Proposed Solution:**
```python
# Cache octave banks to disk
CACHE_DIR = Path("cache/octave_banks")

def get_cached_octave_bank(track_path, sample_rate):
    """Load octave bank from cache if available."""
    cache_file = CACHE_DIR / f"{track_path.stem}.npy"
    if cache_file.exists():
        # Check if audio file was modified since caching
        if cache_file.stat().st_mtime > track_path.stat().st_mtime:
            logger.info(f"Loading cached octave bank: {cache_file}")
            return np.load(cache_file, mmap_mode='r')  # Memory-mapped
    return None

def save_octave_bank_cache(track_path, octave_bank):
    """Save octave bank to cache."""
    cache_file = CACHE_DIR / f"{track_path.stem}.npy"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, octave_bank, allow_pickle=False)

# Checkpoint system for long analysis
def analyze_with_checkpoints(audio_data, checkpoint_dir):
    """Resumable analysis with checkpoints."""
    checkpoint_file = checkpoint_dir / "analysis_progress.json"
    
    if checkpoint_file.exists():
        # Resume from checkpoint
        with open(checkpoint_file) as f:
            progress = json.load(f)
        logger.info(f"Resuming from: {progress['last_completed']}")
    else:
        progress = {"last_completed": None}
    
    # Save checkpoints after each major step
    octave_bank = create_octave_bank(audio_data)
    progress["octave_bank_done"] = True
    save_checkpoint(checkpoint_file, progress)
    
    # Continue with analysis...
```

**Benefits:**
- Skip re-analysis if file unchanged
- Resume long-running analysis
- Shared cache across runs
- Reduced processing time on re-runs

**Challenges:**
- Disk I/O overhead
- Cache invalidation logic
- Storage space management
- Thread safety for shared cache

---

### Priority 5: Streaming/Chunked Processing for Very Large Files 🌊

**Impact:** Process files larger than available memory  
**Difficulty:** Hard  
**Implementation time:** 6-8 hours

**Use Case:** Files >2GB or limited memory systems

**Proposed Solution:**
```python
def process_streaming_chunks(audio_file, chunk_size_samples=44100*60):
    """Process audio in streaming chunks to save memory."""
    # Load and process in chunks
    for i, chunk in enumerate(read_audio_chunks(audio_file, chunk_size_samples)):
        # Create octave bank for this chunk
        chunk_octave = create_octave_bank(chunk)
        
        # Accumulate statistics incrementally
        if i == 0:
            accumulator = initialize_accumulator()
        
        update_accumulator(accumulator, chunk_octave)
        
        # Yield intermediate results
        yield partial_results(accumulator)
    
    return final_results(accumulator)

# Modify to accumulate rather than store all data
class StreamingStatistics:
    """Accumulate statistics without storing all data."""
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_sq = 0
        self.max_val = -np.inf
        self.min_val = np.inf
    
    def update(self, data):
        self.count += len(data)
        self.sum += np.sum(data)
        self.sum_sq += np.sum(data**2)
        self.max_val = max(self.max_val, np.max(data))
        self.min_val = min(self.min_val, np.min(data))
    
    def get_rms(self):
        return np.sqrt(self.sum_sq / self.count)
    
    def get_mean(self):
        return self.sum / self.count
```

**Benefits:**
- Process files larger than RAM
- Constant memory usage
- Works on resource-constrained systems

**Challenges:**
- More complex implementation
- Some operations require full data access
- May need to redesign some algorithms

---

## Recommended Implementation Order

### Phase 1: High-Impact, Medium Effort (Week 1)
1. ✅ Implement parallel octave bank processing (4-8x)
2. ✅ Add float32 option (50% memory reduction)

**Expected result:** 20-40x additional speedup, 50% less memory

### Phase 2: Production Features (Week 2)
3. ✅ Smart caching system
4. ✅ Batch processing improvements

**Expected result:** Near-instant re-analysis, better batch throughput

### Phase 3: Advanced Features (Week 3)
5. ✅ Streaming/chunked processing (if needed)

**Expected result:** Handle any file size

---

## Testing Strategy

### Performance Benchmarks
```python
# Add to tests/
def benchmark_analysis():
    """Measure performance improvements."""
    import time
    from pathlib import Path
    
    test_file = Path("test_tracks/sample.flac")
    
    # Time baseline
    start = time.time()
    analyze_single_track(test_file, ...)
    baseline_time = time.time() - start
    
    # Time with optimizations
    start = time.time()
    analyze_single_track_optimized(test_file, ...)
    optimized_time = time.time() - start
    
    speedup = baseline_time / optimized_time
    print(f"Speedup: {speedup:.1f}x")
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def analyze_single_track(track_path, ...):
    # Will show memory usage line-by-line
    ...
```

### Stress Testing
- Process 100+ tracks in batch
- Test with very large files (>1 hour)
- Test on low-memory systems (4GB RAM)
- Test parallel processing safety

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Parallel processing working on 8-core system
- [ ] 50% memory reduction with float32
- [ ] No accuracy loss compared to baseline
- [ ] All tests passing

### Phase 2 Complete When:
- [ ] Cache system working for re-analysis
- [ ] Batch processing handles 50+ tracks efficiently
- [ ] Cache hit rate >80% on repeated runs
- [ ] Memory growth stable over batch

### Phase 3 Complete When:
- [ ] Can process 2+ hour files on 4GB RAM system
- [ ] Streaming mode produces identical results
- [ ] Progress reporting accurate
- [ ] Resumable processing works

---

## Risk Assessment

| Optimization | Risk Level | Mitigation |
|-------------|------------|------------|
| Parallel processing | Medium | Fallback to sequential on error |
| Float32 | Low | Add precision validation |
| Caching | Low | Cache invalidation logic |
| Batch processing | Medium | Resource limits, error isolation |
| Streaming | High | Extensive testing, validation |

---

## Summary

**Current:** 400-500x faster than original  
**After Phase 1:** 8,000-20,000x faster (parallel × existing)  
**After Phase 2:** Near-instant re-analysis + batch efficiency  
**After Phase 3:** Handle files of any size

**Total improvement potential:** 10,000-20,000x faster for most use cases!

🎯 **Recommendation:** Start with Phase 1 (parallel processing + float32)
This gives the biggest bang for buck with reasonable implementation effort.

