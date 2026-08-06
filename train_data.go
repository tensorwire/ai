package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/tensorwire/tokenizer"
)

// trainExample is one supervised example: a prompt the model conditions on and
// a completion it is trained to produce.
type trainExample struct {
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
}

// TrainCorpus is a token stream plus a per-token supervision mask.
//
// Mask[i] reports whether position i's TARGET (token i+1) contributes to the
// loss. Prompt tokens are false: the model conditions on them but is not
// trained to emit them.
//
// This exists because the trainer previously did `tok.Encode(string(raw))` on a
// flat blob and supervised every position equally. For agent-neo's tool-call
// data — a ~2,700-token system prompt and a ~50-token tool call — that puts
// roughly 98% of the gradient on reproducing the system prompt, which the model
// is never asked to generate. The measured effect is a trainer that optimizes
// almost entirely the wrong objective.
type TrainCorpus struct {
	Tokens []int
	Mask   []bool

	// Supervised counts positions whose target is trained. Reported at load so
	// a corpus that is accidentally 98% prompt is visible immediately rather
	// than after a wasted training run.
	Supervised int
}

// LoadTrainCorpus reads either JSONL prompt/completion pairs or a raw text
// blob, and returns a token stream with its supervision mask.
//
// JSONL is detected by the first non-blank line parsing as an object with a
// "completion" field. Anything else is treated as a raw blob and supervised
// everywhere, preserving the previous behaviour for pretraining-style data.
func LoadTrainCorpus(path string, tok *tokenizer.Tokenizer) (*TrainCorpus, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := string(raw)

	if looksLikeJSONL(text) {
		return loadJSONLCorpus(text, tok)
	}

	// Raw blob: no prompt/completion structure, so every position is
	// supervised. Same as the original behaviour.
	toks := tok.Encode(text)
	mask := make([]bool, len(toks))
	for i := range mask {
		mask[i] = true
	}
	return &TrainCorpus{Tokens: toks, Mask: mask, Supervised: len(toks)}, nil
}

func looksLikeJSONL(text string) bool {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var ex trainExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			return false
		}
		return ex.Completion != ""
	}
	return false
}

func loadJSONLCorpus(text string, tok *tokenizer.Tokenizer) (*TrainCorpus, error) {
	c := &TrainCorpus{}
	var n int

	for ln, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var ex trainExample
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			return nil, fmt.Errorf("line %d: %w", ln+1, err)
		}
		if ex.Completion == "" {
			return nil, fmt.Errorf("line %d: empty completion", ln+1)
		}
		n++

		pt := tok.Encode(ex.Prompt)
		ct := tok.Encode(ex.Completion)

		// The position holding the LAST prompt token is supervised: its target
		// is the first completion token, which the model must learn to emit.
		// Getting this boundary wrong by one is the difference between teaching
		// the model to start a tool call and teaching it nothing.
		for range pt {
			c.Tokens = append(c.Tokens, 0)
			c.Mask = append(c.Mask, false)
		}
		copy(c.Tokens[len(c.Tokens)-len(pt):], pt)
		if len(pt) > 0 {
			c.Mask[len(c.Mask)-1] = true
		}

		for _, t := range ct {
			c.Tokens = append(c.Tokens, t)
			c.Mask = append(c.Mask, true)
		}
		// The final completion token has no next token within this example, so
		// its target would be the next example's first prompt token. Mask it.
		if len(ct) > 0 {
			c.Mask[len(c.Mask)-1] = false
		}
	}

	if n == 0 {
		return nil, fmt.Errorf("no examples found")
	}
	for _, m := range c.Mask {
		if m {
			c.Supervised++
		}
	}
	return c, nil
}

// SupervisedFraction is the share of positions that contribute to the loss.
//
// Report it at load. For tool-call SFT this should be well above the ~2% a flat
// blob would give; a low number means the mask is wrong or the data is
// prompt-heavy, and either way the run is not worth starting.
func (c *TrainCorpus) SupervisedFraction() float64 {
	if len(c.Tokens) == 0 {
		return 0
	}
	return float64(c.Supervised) / float64(len(c.Tokens))
}
