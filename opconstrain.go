package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/tensorwire/tokenizer"
)

// opConstrainer forces generated `"op": "..."` values to be real catalog
// entries by restricting which tokens may be sampled inside that string.
//
// Motivation, measured on granite-4.1-3b against agent-neo's 45-op catalog:
// inventing op names is the DOMINANT failure mode — 6 of 10 failures — with the
// full catalog present in the prompt. Two separate prompt instructions (show the
// wrapper; copy the name character-for-character) fixed call FORMAT completely
// and name accuracy not at all; the hallucinations merely changed shape, e.g.
// `gcpcloud.run` becoming `gcpcloud-logs`.
//
// Repairing names afterwards does not work either: fuzzy-matching resolved 1 of
// 6, and resolving the rest would mean guessing between plausible neighbours.
// That is the worst available outcome, because a wrong-but-valid op runs a real
// query against real infrastructure and returns data that looks authoritative.
//
// Constraining removes the class by construction. Every observed hallucination
// diverges from the catalog at a specific character, and a decoder that can only
// extend a valid prefix cannot take that step.
type opConstrainer struct {
	trie *opTrie
	tok  *tokenizer.Tokenizer

	// tokenText caches the decoded text of each token id. Decoding one token at
	// a time inside the sampling loop would otherwise dominate its cost.
	tokenText []string

	// mask is scratch reused across steps to avoid an allocation per token.
	mask []bool

	// active is true once generation is inside an op string literal.
	active bool
	// emitted is what has been generated inside the current op string.
	emitted string
}

// opTrie answers: given the characters so far, which may come next?
type opTrie struct {
	root *opNode
}

type opNode struct {
	next     map[byte]*opNode
	terminal bool
}

func newOpTrie(ops []string) *opTrie {
	t := &opTrie{root: &opNode{next: map[byte]*opNode{}}}
	for _, op := range ops {
		n := t.root
		for i := 0; i < len(op); i++ {
			c := op[i]
			child, ok := n.next[c]
			if !ok {
				child = &opNode{next: map[byte]*opNode{}}
				n.next[c] = child
			}
			n = child
		}
		n.terminal = true
	}
	return t
}

// walk follows a prefix, reporting the node reached.
func (t *opTrie) walk(prefix string) (*opNode, bool) {
	n := t.root
	for i := 0; i < len(prefix); i++ {
		child, ok := n.next[prefix[i]]
		if !ok {
			return nil, false
		}
		n = child
	}
	return n, true
}

// canExtend reports whether `prefix + s` can still become a catalog op.
func (t *opTrie) canExtend(prefix, s string) bool {
	n, ok := t.walk(prefix)
	if !ok {
		return false
	}
	for i := 0; i < len(s); i++ {
		child, ok := n.next[s[i]]
		if !ok {
			return false
		}
		n = child
	}
	return true
}

// isComplete reports whether prefix is exactly a catalog op.
func (t *opTrie) isComplete(prefix string) bool {
	n, ok := t.walk(prefix)
	return ok && n.terminal
}

func newOpConstrainer(ops []string, tok *tokenizer.Tokenizer) *opConstrainer {
	c := &opConstrainer{trie: newOpTrie(ops), tok: tok}
	if tok != nil {
		n := tok.VocabSize()
		c.tokenText = make([]string, n)
		for i := 0; i < n; i++ {
			c.tokenText[i] = tok.Decode([]int{i})
		}
	}
	return c
}

// opValuePrefix is the exact text that precedes an op value in the tool-call
// JSON the prompt asks for. Matching on it is what tells the constrainer when
// to engage — deliberately narrow, so ordinary prose is never constrained.
const opValuePrefix = `"op": "`

// observe updates the constrainer with the text generated so far this turn.
//
// It engages when the tail of the generated text is `"op": "` and disengages
// when the string closes. Only the region between those points is constrained;
// reasoning, prose and every other argument generate freely.
func (c *opConstrainer) observe(generated string) {
	if !c.active {
		// Tolerate the spacing variants a model actually produces.
		for _, p := range []string{opValuePrefix, `"op":"`, `"op" : "`} {
			if strings.HasSuffix(generated, p) {
				c.active = true
				c.emitted = ""
				return
			}
		}
		return
	}
	// Already inside: recover the emitted portion from the last opening quote.
	if i := strings.LastIndex(generated, `"`); i >= 0 {
		c.emitted = generated[i+1:]
	}
}

// allow reports whether token `id` may be sampled next.
//
// Outside an op string every token is allowed. Inside one, a token is allowed
// when it keeps the op a valid prefix — or when it closes the string and the op
// is already complete, which is what rejects truncations like "pagerduty" (a
// real namespace, not a real op).
func (c *opConstrainer) allow(id int) bool {
	if !c.active || id < 0 || id >= len(c.tokenText) {
		return true
	}
	s := c.tokenText[id]
	if s == "" {
		return true
	}

	// Closing the string is legal only if what we have is a complete op.
	if strings.HasPrefix(s, `"`) {
		return c.trie.isComplete(c.emitted)
	}
	// A token containing a quote mid-way would close the string early; allow it
	// only if the part before the quote completes the op.
	if i := strings.IndexByte(s, '"'); i >= 0 {
		return c.trie.canExtend(c.emitted, s[:i]) &&
			c.trie.isComplete(c.emitted+s[:i])
	}
	return c.trie.canExtend(c.emitted, s)
}

// reset clears per-request state.
func (c *opConstrainer) reset() {
	c.active = false
	c.emitted = ""
}

// constrainLogits masks disallowed tokens in place.
//
// Masking rather than rejection-sampling keeps the change to one pass over the
// logits and leaves the sampler untouched. If the mask would eliminate every
// token — which should be impossible while the prefix is valid, but would
// deadlock generation if it happened — the logits are left alone and the
// constraint yields for that step.
func (c *opConstrainer) constrainLogits(logits []float32) {
	if !c.active {
		return
	}
	const neg = float32(-1e30)
	// One pass: record the verdict, then apply it. Calling allow() twice per
	// token would double the cost of the hot path for no benefit.
	if cap(c.mask) < len(logits) {
		c.mask = make([]bool, len(logits))
	}
	mask := c.mask[:len(logits)]
	kept := 0
	for id := range logits {
		ok := c.allow(id)
		mask[id] = ok
		if ok {
			kept++
		}
	}
	if kept == 0 {
		return
	}
	for id := range logits {
		if !mask[id] {
			logits[id] = neg
		}
	}
}

// loadOpCatalog reads a newline-delimited list of valid op names.
//
// A flat text file rather than a schema: the catalog is generated from
// agent-neo's op registry, and keeping the interchange format trivial means the
// server needs no knowledge of agent-neo's types. Blank lines and # comments
// are ignored so the file can be self-describing.
func loadOpCatalog(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ops []string
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		ops = append(ops, line)
	}
	if len(ops) == 0 {
		return nil, fmt.Errorf("%s: no ops found", path)
	}
	return ops, nil
}
