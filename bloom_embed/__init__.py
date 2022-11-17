import jax.numpy as jnp
import haiku as hk
import numpy as np
import jax.experimental.sparse as jsp
import jax


def mueller_hash(k):
    "https://stackoverflow.com/a/12996028"
    k = ((k >> 16) ^ k) * 0x45D9F3B
    k = ((k >> 16) ^ k) * 0x45D9F3B
    k = (k >> 16) ^ k
    return k


def bloom_ledger(vocab_size, num_digest):
    i = []
    j = []
    ids = np.arange(vocab_size).astype(np.uint32)
    for _ in range(num_digest):
        ids = mueller_hash(ids)
        i.append(ids % vocab_size)
        ids = mueller_hash(ids)
        j.append(ids % vocab_size)
    i, j = map(np.concatenate, (i, j))
    data = np.full((vocab_size * num_digest), 1 / np.sqrt(num_digest))
    indices = np.stack([i, j], axis=-1)
    return jsp.BCOO((data, indices), shape=(vocab_size,) * 2)


def sparse_one_hot(ids, num_categories):
    axes = [range(n) for n in ids.shape[:-1]]
    axes.append(ids.flatten())
    data = np.ones_like(ids)
    indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    mat = jsp.BCOO((data, indices), shape=(*ids.shape, num_categories))
    return mat


class BloomEmbed(hk.Module):
    def __init__(
        self, embed_dim, vocab_size, num_digest, expand_factor=None, name=None
    ):
        super().__init__(name=name)
        self.table = hk.get_parameter(
            "table",
            (vocab_size, embed_dim),
            init=hk.initializers.RandomNormal(1 / np.sqrt(embed_dim)),
        )
        self.bloom = bloom_ledger(vocab_size, num_digest)
        self.ffn = hk.nets.MLP(
            [embed_dim * (expand_factor or num_digest), embed_dim],
            activation=jax.nn.gelu,
        )

    def __call__(self, tokens):
        if jnp.issubdtype(tokens.dtype, jnp.integer):
            tokens = sparse_one_hot(tokens, self.bloom.shape[-1])
        digest = tokens @ self.bloom
        return self.ffn(digest @ self.table)
