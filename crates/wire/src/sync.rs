use crate::types::MemoryChunkDiff;
use std::collections::HashMap;
use xxhash_rust::xxh3::xxh3_64;

pub const CHUNK_SIZE: usize = 64 * 1024; // 64 KB

#[derive(Debug, Default, Clone)]
pub struct MemorySyncTracker {
    chunk_hashes: HashMap<(u64, u64), u64>,
}

impl MemorySyncTracker {
    pub fn new() -> Self {
        Self {
            chunk_hashes: HashMap::new(),
        }
    }

    pub fn compute_dirty_chunks(
        &mut self,
        allocation_id: u64,
        memory: &[u8],
    ) -> Vec<MemoryChunkDiff> {
        let mut diffs = Vec::new();
        let total_len = memory.len();
        let mut offset = 0usize;

        while offset < total_len {
            let chunk_len = (total_len - offset).min(CHUNK_SIZE);
            let chunk_data = &memory[offset..offset + chunk_len];
            let hash = xxh3_64(chunk_data);

            let key = (allocation_id, offset as u64);
            let dirty = match self.chunk_hashes.get(&key) {
                Some(&prev_hash) => prev_hash != hash,
                None => true,
            };

            if dirty {
                self.chunk_hashes.insert(key, hash);
                diffs.push(MemoryChunkDiff {
                    allocation_id,
                    offset: offset as u64,
                    data: chunk_data.to_vec(),
                });
            }

            offset += chunk_len;
        }

        diffs
    }

    /// Like `compute_dirty_chunks`, but only examines the chunks starting at the
    /// given offsets (multiples of `CHUNK_SIZE`, e.g. from OS write tracking).
    pub fn compute_dirty_chunks_at(
        &mut self,
        allocation_id: u64,
        memory: &[u8],
        chunk_offsets: &[usize],
    ) -> Vec<MemoryChunkDiff> {
        let mut diffs = Vec::new();
        for &offset in chunk_offsets {
            if offset >= memory.len() || offset % CHUNK_SIZE != 0 {
                continue;
            }
            let chunk_len = (memory.len() - offset).min(CHUNK_SIZE);
            let chunk_data = &memory[offset..offset + chunk_len];
            let hash = xxh3_64(chunk_data);
            let key = (allocation_id, offset as u64);
            if self.chunk_hashes.get(&key) != Some(&hash) {
                self.chunk_hashes.insert(key, hash);
                diffs.push(MemoryChunkDiff {
                    allocation_id,
                    offset: offset as u64,
                    data: chunk_data.to_vec(),
                });
            }
        }
        diffs
    }

    pub fn mark_clean(&mut self, allocation_id: u64, offset: u64, data: &[u8]) {
        let hash = xxh3_64(data);
        self.chunk_hashes.insert((allocation_id, offset), hash);
    }

    /// Seed the initial baseline hash for clean zeroed allocations so untouched zeros
    /// are not spuriously marked dirty and transmitted on the first frame submission.
    pub fn init_clean_chunks(&mut self, allocation_id: u64, total_len: usize) {
        let zero_chunk = [0u8; CHUNK_SIZE];
        let full_zero_hash = xxh3_64(&zero_chunk);
        let mut offset = 0usize;
        while offset < total_len {
            let chunk_len = (total_len - offset).min(CHUNK_SIZE);
            let hash = if chunk_len == CHUNK_SIZE {
                full_zero_hash
            } else {
                xxh3_64(&zero_chunk[..chunk_len])
            };
            self.chunk_hashes.insert((allocation_id, offset as u64), hash);
            offset += chunk_len;
        }
    }

    pub fn remove_allocation(&mut self, allocation_id: u64) {
        self.chunk_hashes.retain(|&(id, _), _| id != allocation_id);
    }

    pub fn clear(&mut self) {
        self.chunk_hashes.clear();
    }
}
