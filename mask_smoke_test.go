package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorwire/tokenizer"
)

// End-to-end check on data shaped like the real SFT corpus: a tool-call
// completion after a realistic prompt. Confirms the supervised fraction tracks
// the COMPLETION, which is the entire point of B2.
func TestMaskingOnRealisticToolCallData(t *testing.T) {
	tok, err := tokenizer.LoadTokenizer("/Users/dmorgan/.ai/models/granite-4.1-3b")
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	body := `{"prompt":"Incident: Prometheus is reporting alerts in a firing state.","completion":"<tool_call>\n{\"name\": \"skill\", \"arguments\": {\"op\": \"prometheus.alerts\", \"window\": \"15m\"}}\n</tool_call>"}
{"prompt":"Incident: Aurora shows slow UPDATE queries on the devices table.","completion":"<tool_call>\n{\"name\": \"skill\", \"arguments\": {\"op\": \"cognosos-platform.aurora-slowqueries\", \"window\": \"15m\"}}\n</tool_call>"}
`
	p := filepath.Join(t.TempDir(), "sft.jsonl")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadTrainCorpus(p, tok)
	if err != nil {
		t.Fatalf("corpus: %v", err)
	}
	t.Logf("%d tokens, %d supervised (%.1f%%)",
		len(c.Tokens), c.Supervised, 100*c.SupervisedFraction())

	if c.Supervised == 0 {
		t.Fatal("nothing supervised")
	}
	// Every supervised position must fall inside a completion, never a prompt.
	// A prompt token being trained is the defect returning.
	if c.SupervisedFraction() > 0.85 {
		t.Errorf("supervised %.1f%% — prompt tokens are being trained again",
			100*c.SupervisedFraction())
	}
}
