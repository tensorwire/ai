package main

import "testing"

// A stand-in vocabulary. Real tokenizers split op names across many tokens, and
// the constraint has to hold for every split, so the tests drive it at the
// granularity that matters: whole tokens of varying length, including ones that
// straddle the closing quote.
func testConstrainer(t *testing.T, vocab []string) *opConstrainer {
	t.Helper()
	ops := []string{
		"aws.alarms",
		"gcp.logs",
		"gcp.cloud-sql",
		"gcp.cloud-run",
		"db.query",
		"pagerduty.incidents",
		"elasticsearch.health",
		"elasticsearch.indices",
		"cognosos-platform.aurora-slowqueries",
		"cognosos-platform.duress-active-events",
	}
	c := &opConstrainer{trie: newOpTrie(ops), tokenText: vocab}
	return c
}

// The constraint must be inert until generation enters an op string. Everything
// else — reasoning, prose, other arguments — generates freely.
func TestConstrainerIsInertOutsideOpStrings(t *testing.T) {
	c := testConstrainer(t, []string{"anything", "at", "all"})

	c.observe(`<think>The signal implicates the database tier.</think>`)
	if c.active {
		t.Fatal("engaged on prose")
	}
	for id := range c.tokenText {
		if !c.allow(id) {
			t.Errorf("token %d rejected while inactive", id)
		}
	}
}

func TestConstrainerEngagesOnOpValue(t *testing.T) {
	for _, opening := range []string{`{"op": "`, `{"op":"`, `{"op" : "`} {
		c := testConstrainer(t, []string{"x"})
		c.observe(opening)
		if !c.active {
			t.Errorf("did not engage on %q", opening)
		}
	}
}

// The core claim: a token that would take the name off the catalog is not
// sampleable. Each case is a real divergence from the measured baseline.
func TestConstrainerBlocksObservedHallucinations(t *testing.T) {
	cases := []struct {
		emitted string
		token   string
		allowed bool
		why     string
	}{
		// gcpcloud-logs / gcpgcp.cloud-sql: both diverge right after "gcp".
		{"gcp", "cloud", false, "gcp.logs and gcp.cloud-* all need '.' next"},
		{"gcp", "gcp", false, "namespace doubled"},
		{"gcp", ".", true, "the only legal continuation"},

		// aurora-slowqueries became aurora-queries.
		{"cognosos-platform.aurora-", "queries", false, "aurora-slowqueries, not aurora-queries"},
		{"cognosos-platform.aurora-", "slow", true, "on the catalog path"},

		// duress-active-events became duress-activity.
		{"cognosos-platform.duress-activ", "ity", false, "active-events, not activity"},
		{"cognosos-platform.duress-activ", "e", true, "on the catalog path"},

		// prometheus.alarms for prometheus.alerts is the same shape; this
		// catalog uses elasticsearch to make the point.
		{"elasticsearch.", "health", true, "a real op"},
		{"elasticsearch.", "healthy", false, "not a real op"},
	}

	for _, c := range cases {
		con := testConstrainer(t, []string{c.token})
		con.active = true
		con.emitted = c.emitted
		if got := con.allow(0); got != c.allowed {
			t.Errorf("emitted %q + token %q: allow=%v, want %v — %s",
				c.emitted, c.token, got, c.allowed, c.why)
		}
	}
}

// Truncations are the other shape: a valid PREFIX that is not a complete op.
// The constraint catches them by refusing to close the string.
func TestConstrainerRefusesToCloseOnATruncation(t *testing.T) {
	quote := `", `
	cases := []struct {
		emitted string
		canStop bool
	}{
		{"pagerduty", false},          // a namespace, not an op
		{"pagerduty.incidents", true}, // complete
		{"db.quer", false},            // truncation of db.query
		{"db.query", true},            // complete
		{"gcp.cloud-", false},         // ambiguous between -sql and -run
		{"gcp.cloud-sql", true},       // complete
	}
	for _, c := range cases {
		con := testConstrainer(t, []string{quote})
		con.active = true
		con.emitted = c.emitted
		if got := con.allow(0); got != c.canStop {
			t.Errorf("emitted %q: may close = %v, want %v", c.emitted, got, c.canStop)
		}
	}
}

// A token carrying the rest of the name AND the closing quote must be judged on
// both halves at once — otherwise a truncation slips through whenever the
// tokenizer happens to merge the quote into the final token.
func TestConstrainerHandlesTokensThatStraddleTheClosingQuote(t *testing.T) {
	c := testConstrainer(t, []string{`ms"`, `y"`})
	c.active = true

	c.emitted = "aws.alar"
	if !c.allow(0) {
		t.Error(`token 'ms"' should complete aws.alarms and close`)
	}

	c.emitted = "db.quer"
	if !c.allow(1) {
		t.Error(`token 'y"' should complete db.query and close`)
	}

	// But the same shape must be refused when it does not complete an op.
	c.emitted = "aws.ala"
	if c.allow(0) {
		t.Error(`token 'ms"' after "aws.ala" gives aws.alams — must be refused`)
	}
}

// Masking must leave at least one candidate, or generation deadlocks.
func TestConstrainLogitsNeverMasksEverything(t *testing.T) {
	c := testConstrainer(t, []string{"zzz", "qqq", "www"})
	c.active = true
	c.emitted = "aws.alarms" // complete; no token here extends it

	logits := []float32{1, 2, 3}
	c.constrainLogits(logits)

	var survivors int
	for _, v := range logits {
		if v > -1e29 {
			survivors++
		}
	}
	if survivors == 0 {
		t.Error("every token masked — generation would deadlock")
	}
}

func TestConstrainLogitsMasksOnlyDisallowedTokens(t *testing.T) {
	c := testConstrainer(t, []string{".", "cloud", "gcp"})
	c.active = true
	c.emitted = "gcp"

	logits := []float32{1, 1, 1}
	c.constrainLogits(logits)

	if logits[0] <= -1e29 {
		t.Error("'.' masked, but it is the legal continuation")
	}
	if logits[1] > -1e29 || logits[2] > -1e29 {
		t.Error("'cloud'/'gcp' survived, but both leave the catalog")
	}
}

func TestResetClearsState(t *testing.T) {
	c := testConstrainer(t, []string{"x"})
	c.observe(`{"op": "`)
	c.emitted = "gcp"
	c.reset()
	if c.active || c.emitted != "" {
		t.Errorf("reset left active=%v emitted=%q", c.active, c.emitted)
	}
}
