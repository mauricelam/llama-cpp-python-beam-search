"""
Beam search implementation for the "high-level API" of llama_python_cpp.

This implementation uses the internal classes of llama_python_cpp, and may therefore be broken with future versions of
llama_cpp_python.

Usage:

llm = Llama(
      model_path=MODEL,
      n_gpu_layers=-1,
      logits_all=False,
      n_ctx=4096,
      verbose=False,
      temp=0.95,
)

beam_search(
    llm,
    b"Name 10 presidents of the United States",
    beam_width=4,
    branching_factor=3,
)
"""


import llama_cpp
from llama_cpp import Llama
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Iterable, Union
import sys
from utils import swap_stderr


@dataclass
class BeamState:
    logprobs: np.typing.NDArray[np.single]
    tokens: List[int]
    sampler: 'llama_cpp.llama_sampler_p'
    seq_id: int

    def token_str(self, llm, special=True):
        return llm.detokenize(self.tokens, special=special).decode(errors="replace")

    def as_markdown(self, llm, special=True):
        return self.token_str(llm, special=special).replace("\n", "  \n")


@dataclass
class BeamCandidate:
    candidate_logprob: float
    candidate_token: int
    beam: BeamState

    def normalized_score(self):
        """Returns the negative normalized sum of log probabilities.

        If the list is empty, this returns 0, which implies a probability of 1."""
        return -sum(self.beam.logprobs) - self.candidate_logprob

    def token_str(self, llm, special=True):
        return llm.detokenize(
            self.beam.tokens + [self.candidate_token], special=special
        ).decode(errors="replace")


MAX_BATCH_LEN = 128


def _process_prompt_tokens(ctx, prompt_tokens: Sequence[int], *, log=lambda *_: None):
    batch = llama_cpp._internals.LlamaBatch(
        n_tokens=MAX_BATCH_LEN,
        embd=0,
        n_seq_max=1,
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
            batch.batch.n_tokens = n_tokens
            for i in range(n_tokens):
                batch.batch.token[i] = batch_tokens[n_tokens - i - 1]
                batch.batch.pos[i] = b + n_tokens - i - 1
                batch.batch.seq_id[i][0] = 0
                batch.batch.n_seq_id[i] = 1
                batch.batch.logits[i] = False
            batch.batch.logits[0] = True
            ctx.decode(batch)
    finally:
        batch.close()


# def create_ctx(llm, beam_width: int, logfile):
#     """Create a Llama context. The Python Llama class also create a context, but the context param
#     does not support n_seq_max, so we have to create a new one. Also n_ctx needs to be multiplied
#     by the beam width."""
#     with swap_stderr(logfile):
#         ctx_params = llama_cpp.llama_context_default_params()
#         ctx_params.seed = llm._seed
#         ctx_params.n_ctx = llm.n_ctx() * beam_width
#         ctx_params.n_seq_max = beam_width
#         ctx_params.n_batch = max(beam_width, MAX_BATCH_LEN)
#         ctx_params.n_threads = 1
#         ctx_params.n_threads_batch = 1
#         return llama_cpp._internals.LlamaContext(
#             model=llm,
#             params=ctx_params,
#         )


def create_sampler(llm):
    """Create a sampler. This returns a raw C++ pointer directly from the low level API, beacuse
    AFAICT the high level Python samplers do not support cloning."""
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
            llm._model.vocab,
            llama_cpp.llama_n_ctx_train(llm.model),
            0.8,  # DRY multiplier
            1.75,  # DRY base
            4,  # DRY allowed length
            20,  # dry_penalty_last_n
            None,  # seq_breakers
            0,  # num_breakers
        ),
    )
    llama_cpp.llama_sampler_chain_add(
        sampler, llama_cpp.llama_sampler_init_dist(llm._seed)
    )
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


# def allocate_beamed_context(llm, beam_width: int, logfile):
#     # if llm._ctx is not None:
#     #     llm._ctx.close()
#     llm._ctx = llm._stack.enter_context(
#         contextlib.closing(create_ctx(llm, beam_width, logfile))
#     )
#     llm._allocated_beam_ctx = True


def beam_search(
    llm,
    prompt_tokens: Union[bytes, List[int]],
    *,
    n_tokens: int = 512,
    beam_width: int,
    branching_factor: int = 0,
    logits_processors=[],
    logfile=sys.stderr,
):

    def log(*msg):
        print(*msg, file=logfile, flush=True)

    branching_factor = branching_factor or beam_width
    if isinstance(prompt_tokens, bytes):
        prompt_tokens = llm.tokenize(prompt_tokens, special=True)
    log(
        f'Prompt tokens={prompt_tokens}\ndetokenized={[llm.detokenize([tok], special=True).decode(errors="replace") for tok in prompt_tokens]}'
    )
    prompt_tokens_len = len(prompt_tokens)
    # if not getattr(llm, '_allocated_beam_ctx', None):
    #     allocate_beamed_context(llm, beam_width, logfile)
    ctx = llm._ctx
    _process_prompt_tokens(ctx, prompt_tokens, log=log)

    beams = [
        BeamState(
            np.array([]),
            [],
            sampler=create_sampler(llm),
            seq_id=0,
        )
    ]
    terminated_beams = []

    batch = llama_cpp._internals.LlamaBatch(
        n_tokens=beam_width,
        embd=0,
        n_seq_max=beam_width,
    )

    try:
        for i in range(n_tokens):
            log(f"===== Generating token {i} =====")
            effective_beam_width = beam_width - len(terminated_beams)
            log(f"Effective beam width={effective_beam_width}")
            beam_candidates = beam_step(
                llm,
                ctx,
                beams,
                prompt_tokens=prompt_tokens,
                branching_factor=branching_factor,
                n_vocab=llm.n_vocab(),
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
                    ctx.kv_cache_seq_cp(
                        seq_id_src=candidate.beam.seq_id,
                        seq_id_dst=new_seq_id,
                        p0=0,
                        p1=prompt_tokens_len + len(candidate.beam.tokens),
                    )
                new_sampler = llama_cpp.llama_sampler_clone(candidate.beam.sampler)
                llama_cpp.llama_sampler_accept(new_sampler, candidate.candidate_token)
                beam_state = BeamState(
                    np.append(candidate.beam.logprobs, [candidate.candidate_logprob]),
                    candidate.beam.tokens + [candidate.candidate_token],
                    new_sampler,
                    new_seq_id,
                )
                if llama_cpp.llama_token_is_eog(
                    llm._model.vocab, candidate.candidate_token
                ):
                    terminated_beams.append(beam_state)
                else:
                    beams.append(beam_state)
            assert is_unique([b.seq_id for b in beams])

            yield beams + terminated_beams

            if len(beams) <= 0:
                log(f"No more beams to process: {len(beams)}")
                return

            # Convert the beams into the next batch for processing
            batch.batch.n_tokens = len(beams)
            for j, beam in enumerate(beams):
                log(
                    "  "
                    f"Batch: {i} "
                    f"token={beam.tokens[-1]} "
                    f"pos={prompt_tokens_len + len(beam.tokens)} "
                    f"seq_id={beam.seq_id}"
                )
                batch.batch.token[j] = beam.tokens[-1]
                batch.batch.pos[j] = prompt_tokens_len + len(beam.tokens)
                batch.batch.seq_id[j][0] = beam.seq_id
                batch.batch.n_seq_id[j] = 1
                batch.batch.logits[j] = True

            ctx.decode(batch)
            batch.reset()
    finally:
        with swap_stderr(logfile):
            batch.close()
            ctx.close()


def is_unique(seq: Sequence):
    return len(seq) == len(set(seq))


def beam_step(
    llm,
    ctx,
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
        log(f"  Beam {i}/{len(beams)}: {beam.token_str(llm)!r}\n    {beam}")
        logits_ptr = llama_cpp.llama_get_logits_ith(ctx.ctx, i)
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
                debug_detokenize=lambda t: llm.detokenize([t], special=True).decode(
                    errors="replace"
                ),
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
    data_array = llama_cpp._internals.LlamaTokenDataArray(n_vocab=n_vocab)
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
