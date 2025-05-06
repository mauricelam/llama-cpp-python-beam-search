"""
Beam search implementation for the "low-level API" of llama_python_cpp.
The low-level API are direct bindings for the C++ API, and is more stable than the "internal" implementation.
Usage:
params = llama_cpp.llama_model_default_params()
params.n_gpu_layers = 999
model = llama_cpp.llama_load_model_from_file(
    b"/path/to/gguf", params
)
prompt = '''<start_of_turn>user
Name 10 states in the USA<end_of_turn>
<start_of_turn>model
'''.encode()
vocab = llama_cpp.llama_model_get_vocab(model)
with open('gen_log.txt', 'w') as logfile:
    for beams in beam_search(model, prompt, n_ctx=500, beam_width=4, logfile=logfile):
        clear_output(wait=True)
        for beam in beams:
            display(Markdown(beam.as_markdown(vocab)))
"""


import llama_cpp
from llama_cpp import Llama
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Iterable, Union, Optional, Generator
import sys
from utils import swap_stderr
import ctypes


def _vocab_detokenize(
    vocab: 'llama_cpp.llama_vocab_p', tokens: List[int], special=False
) -> bytes:
    output = b""
    size = 32
    buffer = (ctypes.c_char * size)()
    for token in tokens:
        n = llama_cpp.llama_token_to_piece(
            vocab, llama_cpp.llama_token(token), buffer, size, 0, special
        )
        assert n <= size
        output += bytes(buffer[:n])
    # NOTE: Llama1 models automatically added a space at the start of the prompt
    # this line removes a leading space if the first token is a beginning of sentence token
    return (
        output[1:]
        if len(tokens) > 0
        and tokens[0] == llama_cpp.llama_token_bos(vocab)
        and output[0:1] == b" "
        else output
    )


@dataclass
class BeamState:
    logprobs: np.typing.NDArray[np.single]
    tokens: List[int]
    sampler: 'llama_cpp.llama_sampler_p'
    seq_id: int

    def token_str(self, vocab: 'llama_cpp.llama_vocab_p', special=True):
        return _vocab_detokenize(vocab, self.tokens, special).decode(errors="replace")

    def as_markdown(self, vocab: 'llama_cpp.llama_vocab_p', special=True):
        return self.token_str(vocab, special=special).replace("\n", "  \n")


@dataclass
class BeamCandidate:
    candidate_logprob: float
    candidate_token: int
    beam: BeamState

    def normalized_score(self):
        """Returns the negative normalized sum of log probabilities.

        If the list is empty, this returns 0, which implies a probability of 1."""
        return -sum(self.beam.logprobs) - self.candidate_logprob

    def token_str(self, vocab: 'llama_cpp.llama_vocab_p', special=True):
        return _vocab_detokenize(
            vocab, self.beam.tokens + [self.candidate_token], special=special
        ).decode(errors="replace")


MAX_BATCH_LEN = 128


def _process_prompt_tokens(
    ctx: llama_cpp.llama_context_p, prompt_tokens: Sequence[int], *, log=lambda *_: None
) -> None:
    batch = llama_cpp.llama_batch_init(
        MAX_BATCH_LEN,  # n_tokens
        0,  # embd
        1,  # n_seq_max
    )
    try:
        for b in range(0, len(prompt_tokens), MAX_BATCH_LEN):
            batch_tokens = prompt_tokens[b : b + MAX_BATCH_LEN]
            log(f"Processing prompt batch: {b}:{b + MAX_BATCH_LEN}")
            n_tokens = len(batch_tokens)
            # Create the batch in the reverse order from the tokens, so that logits[0] is the one
            # populated, representing the last token in the prompt.
            # This allows us to reuse the same logic for processing the prompt and the subsequent
            # beams, where logits[i] is from the i-th beam
            batch.n_tokens = n_tokens
            for i in range(n_tokens):
                batch.token[i] = batch_tokens[n_tokens - i - 1]
                batch.pos[i] = b + n_tokens - i - 1
                batch.seq_id[i][0] = 0
                batch.n_seq_id[i] = 1
                batch.logits[i] = False
            batch.logits[0] = True
            returncode = llama_cpp.llama_decode(ctx, batch)
            if returncode != 0:
                raise RuntimeError(f"llama_decode failed with code {returncode}")
    finally:
        llama_cpp.llama_batch_free(batch)


def create_ctx(
    model: llama_cpp.llama_model_p, seed: int, n_ctx: int, beam_width: int, logfile
) -> llama_cpp.llama_context_p:
    """Create a Llama context. The Python Llama class also create a context, but the context param
    does not support n_seq_max, so we have to create a new one. Also n_ctx needs to be multiplied
    by the beam width."""
    with swap_stderr(logfile):
        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.seed = seed
        ctx_params.n_ctx = n_ctx * beam_width
        ctx_params.n_seq_max = beam_width
        ctx_params.n_batch = max(beam_width, MAX_BATCH_LEN)
        ctx_params.n_threads = 1
        ctx_params.n_threads_batch = 1
        ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)
        if ctx is None:
            raise RuntimeError("Cannot allocate context")
        return ctx


def create_sampler(
    model: llama_cpp.llama_model_p, vocab: 'llama_cpp.llama_vocab_p', seed: int
) -> 'llama_cpp.llama_sampler_p':
    sparams = llama_cpp.llama_sampler_chain_default_params()
    sampler = llama_cpp.llama_sampler_chain_init(sparams)
    # Add top_k, top_p, temperature, and final distribution-based sampler
    # llama_cpp.llama_sampler_chain_add(sampler, llama_cpp.llama_sampler_init_top_k(40))
    # llama_cpp.llama_sampler_chain_add(
    #     sampler, llama_cpp.llama_sampler_init_top_p(0.9, 1)
    # )
    llama_cpp.llama_sampler_chain_add(sampler, llama_cpp.llama_sampler_init_temp(1.0))
    llama_cpp.llama_sampler_chain_add(
        sampler,
        llama_cpp.llama_sampler_init_dry(
            vocab,
            llama_cpp.llama_n_ctx_train(model),
            0.8,  # DRY multiplier
            1.75,  # DRY base
            4,  # DRY allowed length
            20,  # dry_penalty_last_n
            None,  # seq_breakers
            0,  # num_breakers
        ),
    )
    llama_cpp.llama_sampler_chain_add(sampler, llama_cpp.llama_sampler_init_dist(seed))
    return sampler


class SeqAllocator:
    def __init__(self, n_seq_max: int, used_ids: Iterable[int]):
        self.free_ids = set(range(n_seq_max)) - set(used_ids)
        self.allocated_ids = set()

    def allocate(self, old_seq: int) -> int:
        if old_seq in self.allocated_ids:
            result = self.free_ids.pop()
        else:
            result = old_seq
        self.allocated_ids.add(result)
        return result


def beam_search(
    model: llama_cpp.llama_model_p,
    prompt_tokens: Union[bytes, List[int]],
    n_ctx: Optional[int] = None,
    seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
    ctx: Optional[llama_cpp.llama_context_p] = None,
    *,
    n_tokens: int = 512,
    beam_width: int,
    branching_factor: int = 0,
    logits_processors=[],
    logfile=sys.stderr,
) -> Generator[List[BeamState], None, None]:

    def log(*msg):
        print(*msg, file=logfile, flush=True)

    branching_factor = branching_factor or beam_width
    vocab = llama_cpp.llama_model_get_vocab(model)
    if vocab is None:
        raise RuntimeError("Cannot get vocab from model")
    if isinstance(prompt_tokens, bytes):
        n_ctx_train = llama_cpp.llama_n_ctx_train(model)
        out_tokens = (llama_cpp.llama_token * n_ctx_train)()
        add_bos = True
        special = True
        len_result = llama_cpp.llama_tokenize(
            vocab,
            prompt_tokens,
            len(prompt_tokens),
            out_tokens,
            n_ctx_train,
            add_bos,
            special,
        )
        if len_result < 0:
            raise RuntimeError(f"Failed to tokenize prompt. Result={len_result}")
        prompt_tokens = [out_tokens[i] for i in range(len_result)]
    detokenized = [
        _vocab_detokenize(vocab, [tok], special=True).decode(errors="replace")
        for tok in prompt_tokens
    ]
    log(f"Prompt tokens={prompt_tokens}\ndetokenized={detokenized}")
    prompt_tokens_len = len(prompt_tokens)
    # if not getattr(llm, '_allocated_beam_ctx', None):
    #     allocate_beamed_context(llm, beam_width, logfile)
    allocated_ctx = None
    if ctx is None:
        assert n_ctx is not None
        allocated_ctx = create_ctx(model, seed, n_ctx, beam_width, logfile)
        ctx = allocated_ctx
    _process_prompt_tokens(ctx, prompt_tokens, log=log)

    beams = [
        BeamState(
            np.array([]),
            [],
            sampler=create_sampler(model, vocab, seed),
            seq_id=0,
        )
    ]
    terminated_beams = []

    batch = llama_cpp.llama_batch_init(
        beam_width,  # n_tokens
        0,  # embd
        beam_width,  # n_seq_max
    )

    try:
        n_vocab = llama_cpp.llama_n_vocab(vocab)
        for i in range(n_tokens):
            log(f"===== Generating token {i} =====")
            effective_beam_width = beam_width - len(terminated_beams)
            log(f"Effective beam width={effective_beam_width}")
            beam_candidates = beam_step(
                vocab,
                ctx,
                beams,
                prompt_tokens=prompt_tokens,
                branching_factor=branching_factor,
                n_vocab=n_vocab,
                log=log,
                logits_processors=logits_processors,
            )
            beam_candidates.sort(key=lambda b: b.normalized_score())

            seq_allocator = SeqAllocator(
                beam_width,
                (c.beam.seq_id for c in beam_candidates[:effective_beam_width]),
            )

            beams = []
            for j, candidate in enumerate(beam_candidates):
                if j >= effective_beam_width:
                    log(
                        f"  ❌ Candidate (score={candidate.normalized_score()}): {candidate}"
                    )
                    continue
                else:
                    log(
                        f"  ✅ Candidate (score={candidate.normalized_score()}): {candidate}"
                    )
                new_seq_id = seq_allocator.allocate(candidate.beam.seq_id)
                if new_seq_id != candidate.beam.seq_id:
                    log(f"      Copying seq {candidate.beam.seq_id} to {new_seq_id}")
                    llama_cpp.llama_kv_cache_seq_cp(
                        ctx,
                        candidate.beam.seq_id,  # seq_id_src
                        new_seq_id,  # seq_id_dst
                        0,  # p0
                        prompt_tokens_len + len(candidate.beam.tokens),  # p1
                    )
                new_sampler = llama_cpp.llama_sampler_clone(candidate.beam.sampler)
                llama_cpp.llama_sampler_accept(new_sampler, candidate.candidate_token)
                beam_state = BeamState(
                    np.append(candidate.beam.logprobs, [candidate.candidate_logprob]),
                    candidate.beam.tokens + [candidate.candidate_token],
                    new_sampler,
                    new_seq_id,
                )
                if llama_cpp.llama_token_is_eog(vocab, candidate.candidate_token):
                    terminated_beams.append(beam_state)
                else:
                    beams.append(beam_state)
            assert is_unique([b.seq_id for b in beams])

            yield beams + terminated_beams

            if len(beams) <= 0:
                log(f"No more beams to process: {len(beams)}")
                return

            # Convert the beams into the next batch for processing
            batch.n_tokens = len(beams)
            for j, beam in enumerate(beams):
                log(
                    "  "
                    f"Batch: {i} "
                    f"token={beam.tokens[-1]} "
                    f"pos={prompt_tokens_len + len(beam.tokens)} "
                    f"seq_id={beam.seq_id}"
                )
                batch.token[j] = beam.tokens[-1]
                batch.pos[j] = prompt_tokens_len + len(beam.tokens)
                batch.seq_id[j][0] = beam.seq_id
                batch.n_seq_id[j] = 1
                batch.logits[j] = True

            returncode = llama_cpp.llama_decode(ctx, batch)
            if returncode != 0:
                raise RuntimeError(f"llama_decode failed with code {returncode}")
    finally:
        with swap_stderr(logfile):
            llama_cpp.llama_batch_free(batch)
            if allocated_ctx is not None:
                llama_cpp.llama_free(allocated_ctx)


def is_unique(seq: Sequence):
    return len(seq) == len(set(seq))


def beam_step(
    vocab: 'llama_cpp.llama_vocab_p',
    ctx: llama_cpp.llama_context_p,
    beams: Sequence[BeamState],
    *,
    prompt_tokens: List[int],
    branching_factor: int,
    n_vocab: int,
    logits_processors=[],
    log=lambda *_: None,
) -> List[BeamCandidate]:
    """
    Perform beam search on the given llama_cpp_python LLM.

    Beam search (https://en.wikipedia.org/wiki/Beam_search) is an algorithm that keeps multiple
    best options for the next iteration rather than greedily selecting only one. The number of
    beams can be configured using `num_beams`, trading off between quality (more options
    considered) and performance (higher memory usage and more LLM evaluations). This is especially
    useful when trying to generate in a constrained space, such as with a constrained grammar
    (https://github.com/ggml-org/llama.cpp/issues/2923).

    Yields: A list of the BeamStates, sorted from best to worst.
    """
    beam_candidates: List[BeamCandidate] = []

    # Sample the next token using the sampler chain
    for i, beam in enumerate(beams):
        log(f"  Beam {i}/{len(beams)}: {beam.token_str(vocab)!r}\n    {beam}")
        logits_ptr = llama_cpp.llama_get_logits_ith(ctx, i)
        original_logits = np.ctypeslib.as_array(logits_ptr, (n_vocab,))
        logits = original_logits.copy()
        for lp in logits_processors:
            logits = lp(prompt_tokens + beam.tokens, logits)
        original_logprobs = Llama.logits_to_logprobs(logits.copy()).copy()
        log(f"    logits={logits}")
        for _ in range(branching_factor):
            token = _sample_from_logits(
                beam.sampler,
                logits,
                n_vocab,
                log=log,
                debug_detokenize=lambda t: _vocab_detokenize(
                    vocab, [t], special=True
                ).decode(errors="replace"),
            )

            candidate = BeamCandidate(original_logprobs[token], int(token), beam)
            beam_candidates.append(candidate)
            logits = logits.copy()
            logits[token] = np.float32(-np.inf)

    return beam_candidates


def _sample_from_logits(
    sampler: 'llama_cpp.llama_sampler_p',
    logits: np.typing.NDArray[np.single],
    n_vocab: int,
    *,
    log=lambda *_: None,
    debug_detokenize,
) -> int:
    """Sample a token using the given `sampler` and the given `logits`."""
    # Create data in the shape Array[llama_token_data]
    data_array = LlamaTokenDataArray(n_vocab=n_vocab)
    data_array.copy_logits(logits)
    llama_cpp.llama_sampler_apply(sampler, data_array.candidates)
    selected_data = data_array.candidates.data[data_array.candidates.selected]
    token = selected_data.id
    log(
        f"    Sampled: {token} ({debug_detokenize(token)!r}) "
        f"(logit={selected_data.logit}, p={selected_data.p})"
    )
    log(f"      Candidates after sampling: {data_array.candidates_data.logit[:]}")
    return token


class LlamaTokenDataArray:
    def __init__(self, *, n_vocab: int):
        self.n_vocab = n_vocab
        self.candidates_data = np.recarray(
            (self.n_vocab,),
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self.candidates = llama_cpp.llama_token_data_array(
            data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=self.n_vocab,
            sorted=False,
        )
        self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc)  # type: ignore
        self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)

    def copy_logits(self, logits: np.typing.NDArray[np.single]):
        self.candidates_data.id[:] = self.default_candidates_data_id
        self.candidates_data.logit[:] = logits
        self.candidates_data.p[:] = self.default_candidates_data_p
        self.candidates.sorted = False
        self.candidates.size = self.n_vocab
