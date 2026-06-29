"""The PRISM mind: learns, predicts, discovers ideas, and improves itself."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import string

from letter_follow_mapper import normalize_text
from prism.genome import Genome
from prism.knowledge import Idea, KnowledgeBase

ALPHABET = set(string.ascii_lowercase)
WORD_RE = re.compile(r"[a-z]+")


class PrismMind:
    """
    Self-evolving intelligence layer.

    It ingests text, builds multi-letter transition memory, mines recurring
    patterns as "ideas", and scores itself on prediction quality. Evolution
  picks better genomes (learning strategies) over time.
    """

    def __init__(self, knowledge: KnowledgeBase | None = None) -> None:
        self.knowledge = knowledge or KnowledgeBase()

    @property
    def genome(self) -> Genome:
        return self.knowledge.genome

    def ingest(self, text: str) -> dict[str, int]:
        """Learn from new data. Returns stats about what was absorbed."""
        normalized = normalize_text(text)
        if not normalized:
            return {"bytes": 0, "bigrams": 0, "ideas_added": 0}

        lr = self.genome.learning_rate
        decay = self.genome.memory_decay
        order = self.genome.context_order
        ideas_before = len(self.knowledge.ideas)

        for i in range(len(normalized) - 1):
            context = normalized[max(0, i - order + 1) : i + 1]
            nxt = normalized[i + 1]
            self._update_transition(context, nxt, lr, decay)

        self._discover_ideas(normalized)
        self.knowledge.bytes_ingested += len(text.encode("utf-8"))

        return {
            "bytes": len(text.encode("utf-8")),
            "bigrams": max(0, len(normalized) - 1),
            "ideas_added": len(self.knowledge.ideas) - ideas_before,
        }

    def _update_transition(
        self, context: str, nxt: str, learning_rate: float, decay: float
    ) -> None:
        if nxt not in ALPHABET:
            return
        row = self.knowledge.transitions.setdefault(context, {})
        for key in list(row):
            row[key] *= 1.0 - decay
            if row[key] < 1e-6:
                del row[key]
        row[nxt] = row.get(nxt, 0.0) * (1.0 - learning_rate) + learning_rate
        total = sum(row.values()) or 1.0
        for key in row:
            row[key] /= total

    def _discover_ideas(self, normalized: str) -> None:
        g = self.genome
        ngram_counts: Counter[str] = Counter()
        word_counts: Counter[str] = Counter()

        for length in range(2, 6):
            for i in range(len(normalized) - length + 1):
                ngram_counts[normalized[i : i + length]] += 1

        for match in WORD_RE.finditer(normalized):
            word_counts[match.group()] += 1

        known_patterns = {idea.pattern for idea in self.knowledge.ideas}
        candidates: list[tuple[str, float, str, list[str]]] = []

        for pattern, count in ngram_counts.items():
            if count < g.idea_min_support:
                continue
            rarity = 1.0 / math.sqrt(count)
            strength = count * (1.0 + g.novelty_bonus * rarity)
            followers = self._top_followers(pattern)
            insight = self._insight_for(pattern, count, followers)
            candidates.append((pattern, strength, insight, followers))

        for word, count in word_counts.items():
            if len(word) < 3 or count < g.idea_min_support:
                continue
            strength = count * 1.5
            insight = f"recurring word concept ({count} uses)"
            candidates.append((word, strength, insight, []))

        candidates.sort(key=lambda item: -item[1])
        for pattern, strength, insight, followers in candidates[:40]:
            if pattern in known_patterns:
                self._reinforce_idea(pattern, strength)
                continue
            self.knowledge.ideas.append(
                Idea(
                    pattern=pattern,
                    strength=strength,
                    generation=self.knowledge.generation,
                    connections=followers[:5],
                    insight=insight,
                )
            )
            known_patterns.add(pattern)

        self.knowledge.ideas.sort(key=lambda idea: -idea.strength)
        self.knowledge.ideas = self.knowledge.ideas[:500]

    def _reinforce_idea(self, pattern: str, strength: float) -> None:
        for idea in self.knowledge.ideas:
            if idea.pattern == pattern:
                idea.strength = idea.strength * 0.8 + strength * 0.2
                return

    def _top_followers(self, pattern: str) -> list[str]:
        if not pattern or pattern[-1] not in ALPHABET:
            return []
        row = self.knowledge.transitions.get(pattern[-self.genome.context_order :], {})
        if not row:
            row = self.knowledge.transitions.get(pattern[-1], {})
        ranked = sorted(row.items(), key=lambda item: -item[1])
        return [ch for ch, _ in ranked[:5]]

    def _insight_for(self, pattern: str, count: int, followers: list[str]) -> str:
        if followers:
            return f"pattern '{pattern}' appears {count}x, often followed by {', '.join(followers)}"
        return f"stable pattern '{pattern}' appears {count}x"

    def predict_next(self, context: str) -> dict[str, float]:
        """Return P(next letter | context) using the best available memory."""
        normalized = normalize_text(context)
        if not normalized:
            return {ch: 1 / 26 for ch in string.ascii_lowercase}

        order = self.genome.context_order
        for size in range(min(order, len(normalized)), 0, -1):
            key = normalized[-size:]
            row = self.knowledge.transitions.get(key)
            if row:
                return dict(row)

        return {ch: 1 / 26 for ch in string.ascii_lowercase}

    def evaluate(self, text: str, sample_limit: int = 5000) -> dict[str, float]:
        """
        Score prediction quality on held-out text.

        Higher log-likelihood and top-1 accuracy mean a smarter mind.
        """
        normalized = normalize_text(text)
        if len(normalized) < 3:
            return {"log_likelihood": 0.0, "top1_accuracy": 0.0, "perplexity": 999.0}

        step = max(1, (len(normalized) - 1) // sample_limit)
        log_prob = 0.0
        hits = 0
        total = 0

        for i in range(1, len(normalized) - 1, step):
            context = normalized[:i]
            target = normalized[i]
            dist = self.predict_next(context)
            prob = dist.get(target, 1e-9)
            log_prob += math.log(prob)
            best = max(dist, key=dist.get)
            hits += int(best == target)
            total += 1

        avg_log = log_prob / max(total, 1)
        top1 = hits / max(total, 1)
        perplexity = math.exp(-avg_log)
        return {
            "log_likelihood": avg_log,
            "top1_accuracy": top1,
            "perplexity": perplexity,
        }

    def compute_intelligence(self, eval_stats: dict[str, float]) -> float:
        """
        Composite intelligence score in [0, ~100].

        Grows with prediction skill, idea depth, data absorbed, and generation.
        """
        idea_depth = sum(math.log1p(idea.strength) for idea in self.knowledge.ideas[:50])
        data_factor = math.log1p(self.knowledge.bytes_ingested / 1000)
        prediction = max(0.0, eval_stats.get("top1_accuracy", 0.0)) * 40
        likelihood = max(0.0, eval_stats.get("log_likelihood", -5) + 5) * 8
        evolution = min(20.0, self.knowledge.generation * 0.5)
        score = prediction + likelihood + math.log1p(idea_depth) * 3 + data_factor + evolution
        return round(min(100.0, score), 2)

    def think(self, seed: str, length: int = 40) -> str:
        """Generate text from learned letter dynamics (a creativity probe)."""
        out = normalize_text(seed) or "t"
        for _ in range(length):
            dist = self.predict_next(out)
            nxt = max(dist, key=dist.get)
            out += nxt
        return out

    def status(self) -> dict:
        return {
            "generation": self.knowledge.generation,
            "fitness": round(self.knowledge.fitness, 4),
            "intelligence_score": self.knowledge.intelligence_score,
            "ideas": len(self.knowledge.ideas),
            "transition_contexts": len(self.knowledge.transitions),
            "bytes_ingested": self.knowledge.bytes_ingested,
            "genome": self.knowledge.genome.to_dict(),
        }

    def clone_with_genome(self, genome: Genome) -> PrismMind:
        cloned = KnowledgeBase.from_dict(self.knowledge.to_dict())
        cloned.genome = genome.clamp()
        return PrismMind(cloned)
