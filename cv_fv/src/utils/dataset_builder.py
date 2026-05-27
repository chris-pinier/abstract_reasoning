from pathlib import Path
from typing import *
import json
import math
import warnings
import numpy as np
import pandas as pd
if __name__ == "__main__" and getattr(__builtins__, "__IPYTHON__", False):
    from globals import DATA_DIR; DATA_DIR: Path
else:
    from .globals import DATA_DIR; DATA_DIR: Path
from dataclasses import dataclass

SYMBOLS = [
    "~", "&", ")", "@", "$", "+", "*", "!", "×", '=', '_',
    "-", "/", "}", "]", "€", "±", "^", "%", "¥", '>', '<'
]

WORD_FILES = sorted(DATA_DIR.glob("words_*.txt"))
ALPHABETS = ["symbols"] + [f.stem for f in WORD_FILES]

EXAMPLE_PATTERN = "AABBAABB"
PATTERNS = [
    "AAABAAAB",
    "ABABCDCD",
    "ABBAABBA",
    "ABBACDDC",
    "ABBCABBC",
    "ABCAABCA",
    "ABCDDCBA",
    "ABCDEEDC"
]


class AbstractTask:
    """Generate abstract pattern-completion prompts.

    Each prompt consists of up to three parts, in order:

    1. **(optional) example_pattern** — one solved example using a
       *different* pattern (``example_pattern``), fixed per alphabet.
       Teaches the general task format.
    2. **(optional) n_shot examples** — ``n_shot`` solved examples using
       the *same* pattern as the query, each with fresh symbols.  Freshly
       sampled per prompt.  Teaches the specific pattern.
    3. **query** — the unsolved pattern the model must complete.

    Example (``example_pattern="AABBAABB"``, ``n_shot=1``,
    query pattern ``AAABAAAB``)::

        = = ^ ^ = = ^
        Answer: ^

        + + + ! + + +
        Answer: !

        & & & € & & &
        Answer:

    Symbols for each part are reserved from the pool so they never
    overlap.  When the tokenizer exposes ``apply_chat_template``, all
    parts are wrapped in user/assistant turns automatically.

    Parameters
    ----------
    n : int
        Number of random instantiations per (pattern × alphabet × format).
    n_shot : int
        Solved same-pattern examples before the query (default 0).
    example_pattern : str or None
        Pattern for the introductory example (default ``EXAMPLE_PATTERN``).
        ``None`` disables the introductory example.
    alphabet: symbols | words_eng | words_de | words_es
    q_format, batch_size, tokenizer, seed
        See code for details.
    """

    def __init__(
        self,
        n: int,
        n_shot: int = 0,
        example_pattern: Optional[str] = None,
        alphabet: List[str] = ALPHABETS,
        q_format: List[Literal["open_ended", "multiple_choice"]] = ["open_ended", "multiple_choice"],
        batch_size: Optional[int] = None,
        tokenizer: Any = None,
        system_prompt: str = None,
        seed: int = 42,
    ):
        assert all(a in ALPHABETS for a in alphabet), f"Invalid alphabet: {alphabet}. Valid: {ALPHABETS}"
        assert all(q in ["open_ended", "multiple_choice"] for q in q_format), f"Invalid q_format: {q_format}"

        self.n = n
        self.n_shot = n_shot
        self.example_pattern = example_pattern
        self.alphabet = alphabet
        self.q_format = q_format
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self._chat = (
            tokenizer is not None
            and hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._pools: Dict[str, List[str]] = {"symbols": SYMBOLS}
        for f in WORD_FILES:
            words = [line.strip() for line in open(f) if line.strip()]
            if tokenizer is not None:
                words = [
                    w for w in words
                    if len(tokenizer.tokenize(w)) == 1 and len(tokenizer.tokenize(' ' + w)) == 1
                ]
                # print(f'Dataset {f.stem}; Number of words: {len(words)}')
            self._pools[f.stem] = words

        self._prompts: List[str] = []
        self._display_prompts: List[str] = []
        self._completions: List[str] = []
        self.metadata: List[dict] = []
        self._prompt_data: List[dict] = []
        self._corrupted_prompts: Optional[List[str]] = None
        self._corrupted_display_prompts: Optional[List[str]] = None

        self._build()
        self.batch_size = batch_size or len(self._prompts)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _pool(self, alpha_type: str) -> List[str]:
        return self._pools[alpha_type]

    def _format_prompt(
        self, examples: List[Tuple[str, str]], q_user: str,
    ) -> str:
        """Combine example(s) + query into a single prompt string.

        *examples* is a list of ``(user_content, assistant_content)`` pairs
        (may be empty for 0-shot).  For chat models the content is wrapped
        in user/assistant turns via ``apply_chat_template``; otherwise
        plain-text with newlines.
        """
        if self._chat:
            messages = []
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})
            for ex_user, ex_asst in examples:
                messages.append({"role": "user", "content": ex_user})
                messages.append({"role": "assistant", "content": ex_asst})
            messages.append({"role": "user", "content": q_user})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ) + "Answer:"
        parts = [f"{u}\n{a}" for u, a in examples]
        parts.append(f"{q_user}\nAnswer:")
        return "\n\n".join(parts)

    def _make_options(
        self, answer: str, mapping: Dict[str, str], pool: List[str], n_options: int = 4,
    ) -> Tuple[List[str], int]:
        """Return shuffled MC options and the index of the correct one."""
        used = set(mapping.values())
        distractors = [v for v in used if v != answer]
        remaining = [x for x in pool if x not in used]
        while len(distractors) < n_options - 1:
            pick = remaining[int(self.rng.integers(len(remaining)))]
            distractors.append(pick)
            remaining.remove(pick)
        distractors = list(self.rng.choice(distractors, size=n_options - 1, replace=False))
        options = [answer] + list(distractors)
        self.rng.shuffle(options)
        options = list(options)
        return options, options.index(answer)

    @staticmethod
    def _canonicalize(seq) -> tuple:
        """Map a sequence to its canonical (abstract) form.

        E.g. "ABBA" -> (0,1,1,0), "BAAB" -> (0,1,1,0).
        Two sequences share an abstract pattern iff their canonical forms match.
        """
        seen, counter, result = {}, 0, []
        for ch in seq:
            if ch not in seen:
                seen[ch] = counter
                counter += 1
            result.append(seen[ch])
        return tuple(result)

    def _corrupt_pattern(self, pattern: str) -> str:
        """Randomize the prefix so the abstract pattern is unrecoverable.

        For 8-char patterns the natural midpoint is 4, so positions 0-3 are
        randomized.  For ABCDEEDC all 5 unique symbols must be seen before the
        pattern becomes apparent, so positions 0-4 are randomized.

        Each prefix position is independently assigned a random symbol from the
        pattern's unique set (not just a permutation of the original multiset),
        giving much more variety than simple shuffling.

        Rejects results whose full-sequence canonical form matches the original.
        """
        k = 5 if pattern == "ABCDEEDC" else 4
        pat_uniq = list(dict.fromkeys(pattern))
        suffix = pattern[k:]
        clean_prefix_canon = self._canonicalize(pattern[:k])
        clean_full_canon = self._canonicalize(pattern)

        for _ in range(1000):
            prefix = "".join(
                pat_uniq[i] for i in self._corrupt_rng.integers(len(pat_uniq), size=k)
            )
            corrupted = prefix + suffix
            if corrupted == pattern:
                continue
            if self._canonicalize(corrupted) == clean_full_canon:
                continue
            if self._canonicalize(prefix) == clean_prefix_canon:
                continue
            return corrupted

        raise RuntimeError(f"No valid corruption found for pattern '{pattern}'")

    # ------------------------------------------------------------------
    # dataset construction
    # ------------------------------------------------------------------

    def _build(self):
        for alpha_type in self.alphabet:
            pool = self._pool(alpha_type)

            # --- 1. example_pattern: fixed introductory example (per alphabet) ---
            ep_reserved: set = set()
            ep_data = None
            if self.example_pattern is not None:
                ep_uniq = list(dict.fromkeys(self.example_pattern))
                idxs = self.rng.choice(len(pool), size=len(ep_uniq), replace=False)
                ep_map = {ch: pool[int(i)] for ch, i in zip(ep_uniq, idxs)}
                ep_seq = [ep_map[ch] for ch in self.example_pattern]
                ep_data = (" ".join(ep_seq[:-1]), ep_seq[-1], ep_map)
                ep_reserved.update(ep_map.values())

            base_pool = [w for w in pool if w not in ep_reserved]

            ep_pair: Dict[str, Optional[Tuple[str, str]]] = {}
            if ep_data is not None:
                line, ans, ep_map = ep_data
                ep_pair["open_ended"] = (line, f"Answer: {ans}")
                options, _ = self._make_options(ans, ep_map, pool)
                ep_pair["multiple_choice"] = (
                    f"{line}\nOptions:\n {' '.join(options)}",
                    f"Answer: {ans}",
                )
            else:
                ep_pair["open_ended"] = None
                ep_pair["multiple_choice"] = None

            # --- 2 & 3. Build prompts (n_shot examples freshly sampled per prompt) ---
            for fmt in self.q_format:
                for pattern in PATTERNS:
                    pat_uniq = list(dict.fromkeys(pattern))
                    n_unique = len(pat_uniq)
                    needed = n_unique * (self.n_shot + 1)
                    if len(base_pool) < needed:
                        continue

                    for _ in range(self.n):
                        shot_reserved: set = set()
                        shots: List[Tuple[str, str, Dict[str, str]]] = []
                        for _ in range(self.n_shot):
                            available = [w for w in base_pool if w not in shot_reserved]
                            idxs = self.rng.choice(len(available), size=n_unique, replace=False)
                            s_map = {ch: available[int(i)] for ch, i in zip(pat_uniq, idxs)}
                            s_seq = [s_map[ch] for ch in pattern]
                            shots.append((" ".join(s_seq[:-1]), s_seq[-1], s_map))
                            shot_reserved.update(s_map.values())

                        all_examples: List[Tuple[str, str]] = []
                        shot_options_list: List[Optional[list]] = []
                        if ep_pair[fmt] is not None:
                            all_examples.append(ep_pair[fmt])
                        for s_line, s_ans, s_map in shots:
                            if fmt == "open_ended":
                                all_examples.append((s_line, f"Answer: {s_ans}"))
                                shot_options_list.append(None)
                            else:
                                options, _ = self._make_options(s_ans, s_map, base_pool)
                                all_examples.append((
                                    f"{s_line}\nOptions:\n {'\n '.join(options)}",
                                    f"Answer: {s_ans}",
                                ))
                                shot_options_list.append(list(options))

                        query_pool = [w for w in base_pool if w not in shot_reserved]
                        idxs = self.rng.choice(
                            len(query_pool), size=len(pat_uniq), replace=False,
                        )
                        mapping = {
                            ch: query_pool[int(i)]
                            for ch, i in zip(pat_uniq, idxs)
                        }
                        seq = [mapping[ch] for ch in pattern]
                        answer = seq[-1]
                        q_line = " ".join(seq[:-1])

                        if fmt == "open_ended":
                            q_user = q_line
                            query_options = None
                        else:
                            options, _ = self._make_options(
                                answer, mapping, query_pool,
                            )
                            q_user = f"{q_line}\nOptions:\n {'\n '.join(options)}"
                            query_options = list(options)

                        self._prompts.append(
                            self._format_prompt(all_examples, q_user)
                        )
                        display_parts = [f"{u}\n{a}" for u, a in all_examples]
                        display_parts.append(f"{q_user}\nAnswer:")
                        self._display_prompts.append("\n\n".join(display_parts))
                        self._completions.append(f" {answer}")

                        self.metadata.append({
                            "pattern": pattern,
                            "alphabet": alpha_type,
                            "format": fmt,
                            "mapping": mapping,
                        })
                        self._prompt_data.append({
                            "ep_pair": ep_pair[fmt],
                            "shots": list(shots),
                            "shot_options": shot_options_list,
                            "query_options": query_options,
                        })
    
    @dataclass
    class Prompts:
        x: List[str]
        y: List[str]
        display: List[str]
        pattern: List[str]
        alphabet: List[str]
        format: List[str]
        corrupted_x: Optional[List[str]] = None
        corrupted_display: Optional[List[str]] = None

        _B = "\033[1m"       # bold
        _D = "\033[2m"       # dim
        _R = "\033[0m"       # reset
        _C = "\033[36m"      # cyan
        _Y = "\033[33m"      # yellow
        _G = "\033[32m"      # green
        _M = "\033[35m"      # magenta

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx: int) -> str:
            header = (
                f"{self._D}[{idx}]{self._R}  "
                f"{self._C}alphabet={self._B}{self.alphabet[idx]}{self._R}  "
                f"{self._Y}format={self._B}{self.format[idx]}{self._R}  "
                f"{self._M}pattern={self._B}{self.pattern[idx]}{self._R}"
            )
            print(f"{header}\n{self.display[idx]}\n{self._G}y = {self._R}`{self._G}{self._B}{self.y[idx]}{self._R}`")

        def __repr__(self) -> str:
            title = f"{self._G}{self._B}Prompts{self._R}{self._D}(n={len(self.x)}){self._R}"
            lines = [title]
            for i in range(min(3, len(self.x))):
                lines.append(self[i])
            if len(self.x) > 3:
                lines.append(f"{self._D}  ... ({len(self.x) - 3} more){self._R}")
            "\n\n".join(lines)

        @property
        def names(self) -> List[str]:
            return [
                f"{self.pattern[i]}-{self.alphabet[i]}-{'oe' if self.format[i] == 'open_ended' else 'mc'}" 
                for i in range(len(self.x))
            ]
        

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    @property
    def prompts(self) -> "AbstractTask.Prompts":
        return self.Prompts(
            x=self._prompts,
            y=self._completions,
            display=self._display_prompts,
            pattern=[m["pattern"] for m in self.metadata],
            alphabet=[m["alphabet"] for m in self.metadata],
            format=[m["format"] for m in self.metadata],
            corrupted_x=self._corrupted_prompts,
            corrupted_display=self._corrupted_display_prompts,
        )

    def create_corrupted_dataset(self) -> None:
        """Generate corrupted (prefix-shuffled) prompts for activation patching.

        Each n-shot example and the query get independent prefix shuffles that
        destroy the abstract pattern while keeping the same symbols.
        The example_pattern (if any) is left clean.

        After calling this, iterating the dataset yields
        ``((clean_prompts, completions), (corrupted_prompts, completions))``
        per batch.
        """
        self._corrupt_rng = np.random.default_rng(self.seed ^ 0xDEAD)
        self._corrupted_prompts = []
        self._corrupted_display_prompts = []

        for meta, pdata in zip(self.metadata, self._prompt_data):
            pattern = meta["pattern"]
            mapping = meta["mapping"]
            fmt = meta["format"]
            ep_pair = pdata["ep_pair"]
            shots = pdata["shots"]
            shot_options = pdata["shot_options"]
            query_options = pdata["query_options"]

            corrupted_examples: List[Tuple[str, str]] = []
            if ep_pair is not None:
                corrupted_examples.append(ep_pair)

            for (s_line, s_ans, s_map), s_opts in zip(shots, shot_options):
                c_pat = self._corrupt_pattern(pattern)
                c_s_seq = [s_map[ch] for ch in c_pat]
                c_s_line = " ".join(c_s_seq[:-1])
                c_s_ans = c_s_seq[-1]
                if fmt == "open_ended":
                    corrupted_examples.append((c_s_line, f"Answer: {c_s_ans}"))
                else:
                    corrupted_examples.append((
                        f"{c_s_line}\nOptions:\n {'\n '.join(s_opts)}",
                        f"Answer: {c_s_ans}",
                    ))

            corrupted_pat = self._corrupt_pattern(pattern)
            corrupted_seq = [mapping[ch] for ch in corrupted_pat]
            corrupted_q_line = " ".join(corrupted_seq[:-1])

            if fmt == "open_ended":
                corrupted_q_user = corrupted_q_line
            else:
                corrupted_q_user = f"{corrupted_q_line}\nOptions:\n {'\n '.join(query_options)}"

            self._corrupted_prompts.append(
                self._format_prompt(corrupted_examples, corrupted_q_user)
            )
            corrupted_display = [f"{u}\n{a}" for u, a in corrupted_examples]
            corrupted_display.append(f"{corrupted_q_user}\nAnswer:")
            self._corrupted_display_prompts.append("\n\n".join(corrupted_display))

    def __len__(self):
        return math.ceil(len(self._prompts) / self.batch_size)

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self._prompts))
        if start >= len(self._prompts):
            raise IndexError
        clean = (self._prompts[start:end], self._completions[start:end])
        if self._corrupted_prompts is not None:
            corrupted = (self._corrupted_prompts[start:end], self._completions[start:end])
            return clean, corrupted
        return clean

    def get_config(self) -> dict:
        """Return a JSON-serialisable dict that fully describes this dataset's configuration."""
        return {
            "n": self.n,
            "n_shot": self.n_shot,
            "example_pattern": self.example_pattern,
            "alphabet": self.alphabet,
            "q_format": self.q_format,
            "batch_size": self.batch_size if self.batch_size != len(self._prompts) else None,
            "system_prompt": self.system_prompt,
            "seed": self.seed,
            "used_tokenizer": self.tokenizer is not None,
        }

    def save_config(self, path: Union[str, Path]) -> Path:
        """Save dataset config to a JSON file.

        The saved config can be loaded with ``AbstractTask.from_config`` to
        reconstruct the identical dataset (given the same tokenizer).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_config(), f, indent=2)
        return path

    @classmethod
    def from_config(
        cls,
        path: Union[str, Path],
        tokenizer: Any = None,
    ) -> "AbstractTask":
        """Reconstruct an AbstractTask from a saved config JSON."""
        with open(path) as f:
            cfg = json.load(f)
        if cfg.pop("used_tokenizer", False) and tokenizer is None:
            warnings.warn(
                "The original dataset was built with a tokenizer (used for "
                "word-pool filtering and chat templates). Reconstructing "
                "without one will produce a different dataset.",
                stacklevel=2,
            )
        return cls(tokenizer=tokenizer, **cfg)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.metadata)
        df["prompt"] = self._prompts
        if self._corrupted_prompts is not None:
            df["corrupted_prompt"] = self._corrupted_prompts
        df["completion"] = self._completions
        return df
