"""Model-independent, reproducible epoch permutations of fixed token samples."""
import glob
import hashlib
from pathlib import Path

import numpy as np
import torch


class EpochSampleDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes,
                 max_shards=0, data_seed=20261012, shuffle=True, device='cuda'):
        if B <= 0 or T <= 0 or num_processes <= 0 or not 0 <= process_rank < num_processes:
            raise ValueError('invalid batch, sequence length or rank configuration')
        if data_seed < 0:
            raise ValueError('data_seed must be nonnegative')
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        self.data_seed, self.shuffle, self.device = data_seed, bool(shuffle), device
        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise ValueError(f'no shards matched {filename_pattern}')
        if max_shards < 0 or max_shards > len(self.files):
            raise ValueError(f'requested {max_shards} shards; available: {len(self.files)}')
        if max_shards:
            self.files = self.files[:max_shards]
        self.shards, self.shard_metadata = [], []
        for filename in self.files:
            with open(filename, 'rb') as f:
                header = np.frombuffer(f.read(1024), dtype='<i4')
            if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
                raise ValueError(f'invalid FineWeb binary header: {filename}')
            ntok = int(header[2])
            if ntok < T + 1 or Path(filename).stat().st_size != 1024 + ntok * 2:
                raise ValueError(f'invalid or too short shard: {filename}')
            # Samples start at 0,T,2T,...; each needs T+1 tokens for shifted labels.
            nsamples = (ntok - 1) // T
            digest = hashlib.sha256()
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(chunk)
            self.shards.append(np.memmap(filename, mode='r', dtype='<u2', offset=1024, shape=(ntok,)))
            self.shard_metadata.append(dict(file=filename, tokens=ntok, samples=nsamples,
                                           unused_input_tokens=ntok - nsamples * T,
                                           sha256=digest.hexdigest()))
        self.sample_ends = np.cumsum([r['samples'] for r in self.shard_metadata], dtype=np.int64)
        self.sample_starts = np.concatenate(([0], self.sample_ends[:-1]))
        self.num_samples = int(self.sample_ends[-1])
        self.ntok_total = sum(r['tokens'] for r in self.shard_metadata)
        self.usable_tokens_per_epoch = self.num_samples * self.T
        self.reset()

    def _set_epoch(self):
        if self.shuffle:
            # Dedicated RNG: independent of model width, initialization or global RNG use.
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.data_seed, self.epoch])))
            self.order = rng.permutation(self.num_samples)
        else:
            self.order = np.arange(self.num_samples, dtype=np.int64)

    def reset(self):
        self.epoch, self.position = 0, 0
        self._set_epoch()

    def close(self):
        for shard in self.shards:
            shard._mmap.close()

    def next_sample_ids(self):
        remaining = self.B * self.num_processes
        pieces = []
        while remaining:
            take = min(remaining, self.num_samples - self.position)
            pieces.append(self.order[self.position:self.position + take])
            self.position += take
            remaining -= take
            if self.position == self.num_samples:
                self.epoch += 1
                self.position = 0
                self._set_epoch()
        global_ids = np.concatenate(pieces)
        start = self.process_rank * self.B
        return global_ids[start:start + self.B]

    def next_batch(self):
        ids = self.next_sample_ids()
        self.last_sample_ids = ids.copy()
        shard_ids = np.searchsorted(self.sample_ends, ids, side='right')
        tokens = np.empty((self.B, self.T + 1), dtype=np.int32)
        for row, (sample_id, shard_id) in enumerate(zip(ids, shard_ids)):
            offset = int(sample_id - self.sample_starts[shard_id]) * self.T
            tokens[row] = self.shards[shard_id][offset:offset + self.T + 1]
        # Each sample retains its original internal token order and next-token labels.
        x = torch.tensor(tokens[:, :-1].copy(), dtype=torch.long, device=self.device)
        y = torch.tensor(tokens[:, 1:].copy(), dtype=torch.long, device=self.device)
        return x, y

    def order_metadata(self):
        fingerprint = hashlib.sha256()
        for row in self.shard_metadata:
            fingerprint.update(bytes.fromhex(row['sha256']))
        # Always report epoch 0 even if a batch was prefetched before metadata writing.
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.data_seed, 0])))
        initial = rng.permutation(self.num_samples) if self.shuffle else np.arange(self.num_samples)
        return dict(data_seed=self.data_seed, shuffle=self.shuffle,
                    algorithm='PCG64(SeedSequence([data_seed, epoch])).permutation',
                    numpy_version=np.__version__, sequence_length=self.T,
                    samples_per_epoch=self.num_samples, usable_tokens_per_epoch=self.usable_tokens_per_epoch,
                    shard_content_fingerprint=fingerprint.hexdigest(), shards=self.shard_metadata,
                    epoch0_order_sha256=hashlib.sha256(initial.astype('<i8').tobytes()).hexdigest(),
                    epoch0_first_sample_ids=initial[:16].tolist(),
                    epoch_boundary='carry partial global batch into the next epoch; no samples dropped')
