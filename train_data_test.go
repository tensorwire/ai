package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorwire/tokenizer"
)

func testTok(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.LoadTokenizer("/Users/dmorgan/.ai/models/granite-4.1-3b")
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	return tok
}

func writeTemp(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// The defect this fixes: a 2,700-token prompt with a ~50-token completion put
// ~98% of the gradient on reproducing the system prompt. Masking must invert
// that — the supervised fraction should track completion length, not prompt
// length.
func TestPromptTokensAreNotSupervised(t *testing.T) {
	tok := testTok(t)
	long := ""
	for i := 0; i < 200; i++ {
		long += "the system prompt describes many operations. "
	}
	p := writeTemp(t, "d.jsonl", `{"prompt":`+jsonStr(long)+`,"completion":"skill"}`+"\n")

	c, err := LoadTrainCorpus(p, tok)
	if err != nil {
		t.Fatal(err)
	}
	frac := c.SupervisedFraction()
	t.Logf("%d tokens, %d supervised (%.1f%%)", len(c.Tokens), c.Supervised, 100*frac)

	if frac > 0.10 {
		t.Errorf("supervised fraction %.1f%% — prompt tokens are still being trained", 100*frac)
	}
	if c.Supervised == 0 {
		t.Error("nothing supervised; the completion must be trained")
	}
}

// The boundary is the whole point: the position holding the LAST prompt token
// has the FIRST completion token as its target, so it MUST be supervised.
// Off-by-one here is the difference between teaching the model to begin a tool
// call and teaching it nothing.
func TestLastPromptPositionIsSupervised(t *testing.T) {
	tok := testTok(t)
	p := writeTemp(t, "d.jsonl", `{"prompt":"alpha beta","completion":"gamma delta"}`+"\n")

	c, err := LoadTrainCorpus(p, tok)
	if err != nil {
		t.Fatal(err)
	}
	pt := tok.Encode("alpha beta")
	if len(pt) == 0 {
		t.Skip("tokenizer produced no prompt tokens")
	}
	last := len(pt) - 1
	if !c.Mask[last] {
		t.Errorf("position %d (last prompt token) is masked; its target is the "+
			"first completion token and must be trained", last)
	}
	if last > 0 && c.Mask[last-1] {
		t.Errorf("position %d is supervised but its target is still a prompt token", last-1)
	}
}

// The last completion token's target belongs to the NEXT example. Training it
// teaches the model to run one example into the next.
func TestFinalCompletionTokenIsMasked(t *testing.T) {
	tok := testTok(t)
	p := writeTemp(t, "d.jsonl",
		`{"prompt":"a","completion":"b c"}`+"\n"+`{"prompt":"d","completion":"e f"}`+"\n")

	c, err := LoadTrainCorpus(p, tok)
	if err != nil {
		t.Fatal(err)
	}
	if c.Mask[len(c.Mask)-1] {
		t.Error("final position supervised; its target is out of corpus")
	}
}

// A raw blob has no prompt/completion structure, so it must stay fully
// supervised — this path is what pretraining-style data uses.
func TestRawBlobStaysFullySupervised(t *testing.T) {
	tok := testTok(t)
	p := writeTemp(t, "d.txt", "just some ordinary training text with no structure")

	c, err := LoadTrainCorpus(p, tok)
	if err != nil {
		t.Fatal(err)
	}
	if c.SupervisedFraction() != 1.0 {
		t.Errorf("raw blob supervised fraction %.2f, want 1.0", c.SupervisedFraction())
	}
}

func TestMalformedJSONLIsAnError(t *testing.T) {
	tok := testTok(t)
	p := writeTemp(t, "d.jsonl", `{"prompt":"a","completion":"b"}`+"\n"+`{"prompt":"c"}`+"\n")
	if _, err := LoadTrainCorpus(p, tok); err == nil {
		t.Error("an example with no completion was accepted")
	}
}

func jsonStr(s string) string {
	out := []byte{'"'}
	for _, r := range s {
		switch r {
		case '"':
			out = append(out, '\\', '"')
		case '\\':
			out = append(out, '\\', '\\')
		case '\n':
			out = append(out, '\\', 'n')
		default:
			out = append(out, []byte(string(r))...)
		}
	}
	return string(append(out, '"'))
}
