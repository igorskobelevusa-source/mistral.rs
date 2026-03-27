use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct RotatingCache {
    pub all_data: Option<Tensor>,
    pub dim: usize,
    // `offset` is the current write index in the buffer
    pub offset: usize,
    // The total size of the sequence seen so far.
    pub current_seq_len: usize,
    // max_seq_len is the size of the rotating buffer, it is actually allowed for the full
    // sequence to grow past this limit.
    pub max_seq_len: usize,
    pub capacity_seq_len: usize,
}

impl RotatingCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            offset: 0,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => {
                if self.current_seq_len >= self.max_seq_len {
                    Some(d.clone())
                } else {
                    Some(d.narrow(self.dim, 0, self.current_seq_len)?)
                }
            }
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        // If the buffer has wrapped, the circular data layout is incompatible
        // with a rollback — positions no longer match their original linear
        // indices. Reject so the prefix cacher falls back to full recomputation.
        if self.current_seq_len > self.max_seq_len && len < self.current_seq_len {
            candle_core::bail!(
                "Rotating KV cache cannot roll back a wrapped buffer \
                 (current_seq_len {} > max_seq_len {}, requested len {})",
                self.current_seq_len,
                self.max_seq_len,
                len,
            );
        }
        // If trying to roll it back past the boundary of max_seq_len, fail early.
        if self.current_seq_len.saturating_sub(len) > self.max_seq_len {
            candle_core::bail!(
                "Rotating KV cache (usually for sliding window) tried to reset to len {len} while current is {} and max retained is {}",
                self.current_seq_len,
                self.max_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        self.offset = len % self.max_seq_len;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(self.dim)?;
        // Pre-allocate to max_seq_len on first use to avoid repeated reallocation
        // This matches llama.cpp's approach of allocating full context upfront
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            // Pre-allocate to max_seq_len (sliding window size) instead of capacity_seq_len
            // This eliminates all future reallocations during decode
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
            self.capacity_seq_len = self.max_seq_len;
        };

        let ad = self.all_data.as_mut().unwrap();

        self.current_seq_len += seq_len;
        if seq_len >= self.max_seq_len {
            let narrowed = src.narrow(self.dim, seq_len - self.max_seq_len, self.max_seq_len)?;
            // Only call contiguous if needed
            let to_copy = if narrowed.is_contiguous() { narrowed } else { narrowed.contiguous()? };
            ad.slice_set(&to_copy, self.dim, 0)?;
            self.offset = 0;
            // Here we return `src` rather than `ad` so that all the past can be used.
            Ok(src.clone())
        } else {
            let rem_len = self.max_seq_len - self.offset;
            if seq_len <= rem_len {
                // Only call contiguous if needed - during decode src is usually already contiguous
                let src_contig = if src.is_contiguous() { src.clone() } else { src.contiguous()? };
                ad.slice_set(&src_contig, self.dim, self.offset)?;
                self.offset = (self.offset + seq_len) % self.max_seq_len;
            } else {
                // We have to make two copies here as we go over the boundary of the cache.
                if rem_len > 0 {
                    let src1 = src.narrow(self.dim, 0, rem_len)?;
                    let src1_contig = if src1.is_contiguous() { src1 } else { src1.contiguous()? };
                    ad.slice_set(&src1_contig, self.dim, self.offset)?;
                }
                let src2 = src.narrow(self.dim, rem_len, seq_len - rem_len)?;
                let src2_contig = if src2.is_contiguous() { src2 } else { src2.contiguous()? };
                ad.slice_set(&src2_contig, self.dim, 0)?;
                self.offset = seq_len - rem_len;
            }
            if self.current_seq_len >= self.max_seq_len {
                Ok(ad.clone())
            } else {
                Ok(ad.narrow(self.dim, 0, self.current_seq_len)?)
            }
        }
    }
}
