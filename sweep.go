package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/open-ai-org/mongoose"
)

// cmdSweep runs a hyperparameter search over training configurations.
//
// Default: random search over learning rate (5 values) with 500 steps each.
// Reports the best configuration by final loss.
//
// Usage:
//
//	ai sweep data=corpus.txt                               5 random LR trials
//	ai sweep data=corpus.txt lr=1e-4,3e-4,6e-4            Grid over specific LRs
//	ai sweep data=corpus.txt lr=1e-4,6e-4 dim=64,128      Grid over LR x dim
//	ai sweep data=corpus.txt --trials 10 --steps 200      Random search, 10 trials
func cmdSweep(args map[string]string) {
	dataPath := args["data"]
	if dataPath == "" {
		dataPath = args["_0"]
	}
	if dataPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ai sweep data=<file> [lr=...] [dim=...] [layers=...] [--trials N] [--steps N]")
		os.Exit(1)
	}

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	// Parse search space
	lrs := parseFloatList(args["lr"], []float64{1e-4, 3e-4, 6e-4, 1e-3, 3e-3})
	dims := parseIntList(args["dim"], []int{128})
	layersList := parseIntList(args["layers"], []int{4})
	ffnMults := parseIntList(args["ffn"], nil) // nil = auto (4x dim)

	stepsPerTrial := 500
	if v, ok := args["steps"]; ok {
		fmt.Sscanf(v, "%d", &stepsPerTrial)
	}
	maxTrials := 0
	if v, ok := args["trials"]; ok {
		fmt.Sscanf(v, "%d", &maxTrials)
	}

	// Build configurations
	type trialConfig struct {
		lr     float64
		dim    int
		layers int
		ffnDim int
	}

	var configs []trialConfig
	for _, lr := range lrs {
		for _, dim := range dims {
			for _, layers := range layersList {
				ffnDim := dim * 4
				if len(ffnMults) > 0 {
					for _, f := range ffnMults {
						configs = append(configs, trialConfig{lr, dim, layers, f})
					}
				} else {
					configs = append(configs, trialConfig{lr, dim, layers, ffnDim})
				}
			}
		}
	}

	// Limit trials if requested
	if maxTrials > 0 && len(configs) > maxTrials {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		rng.Shuffle(len(configs), func(i, j int) { configs[i], configs[j] = configs[j], configs[i] })
		configs = configs[:maxTrials]
	}

	fmt.Println("ai sweep — hyperparameter search")
	fmt.Printf("  data:     %s (%s)\n", dataPath, formatBytes(len(raw)))
	fmt.Printf("  trials:   %d\n", len(configs))
	fmt.Printf("  steps:    %d per trial\n", stepsPerTrial)
	fmt.Printf("  search:   lr=%v dim=%v layers=%v\n", lrs, dims, layersList)
	fmt.Println()

	eng := selectEngine("auto")
	graph := mongoose.AsGraphTrainEngine(eng)

	type trialResult struct {
		config   trialConfig
		loss     float32
		elapsed  time.Duration
		stepsPS  float64
	}

	var results []trialResult

	for trial, cfg := range configs {
		fmt.Printf("[%d/%d] lr=%.1e dim=%d layers=%d ffn=%d ... ",
			trial+1, len(configs), cfg.lr, cfg.dim, cfg.layers, cfg.ffnDim)

		loss, elapsed := runSweepTrial(eng, graph, data, dataPath, cfg.lr, cfg.dim, cfg.layers, cfg.ffnDim, stepsPerTrial)
		stepsPS := float64(stepsPerTrial) / elapsed.Seconds()

		results = append(results, trialResult{cfg, loss, elapsed, stepsPS})
		fmt.Printf("loss=%.4f (%.1f steps/s, %v)\n", loss, stepsPS, elapsed.Round(time.Millisecond))
	}

	// Sort by loss
	sort.Slice(results, func(i, j int) bool { return results[i].loss < results[j].loss })

	fmt.Println()
	fmt.Println("Results (sorted by loss):")
	fmt.Printf("  %-6s %-10s %-5s %-7s %-6s %-10s %-10s\n",
		"Rank", "LR", "Dim", "Layers", "FFN", "Loss", "Speed")
	fmt.Printf("  %-6s %-10s %-5s %-7s %-6s %-10s %-10s\n",
		"----", "--", "---", "------", "---", "----", "-----")

	for i, r := range results {
		marker := " "
		if i == 0 {
			marker = "★"
		}
		fmt.Printf("  %-6s %-10.1e %-5d %-7d %-6d %-10.4f %.1f steps/s\n",
			fmt.Sprintf("#%d%s", i+1, marker), r.config.lr, r.config.dim, r.config.layers, r.config.ffnDim,
			r.loss, r.stepsPS)
	}

	best := results[0]
	fmt.Printf("\nBest: lr=%.1e dim=%d layers=%d ffn=%d → loss=%.4f\n",
		best.config.lr, best.config.dim, best.config.layers, best.config.ffnDim, best.loss)
	fmt.Printf("\nRun with: ai train data=%s lr=%.1e dim=%d layers=%d ffn-dim=%d\n",
		dataPath, best.config.lr, best.config.dim, best.config.layers, best.config.ffnDim)
}

func runSweepTrial(eng mongoose.Engine, graph mongoose.GraphTrainEngine, data []int, dataPath string,
	lr float64, dim, layers, ffnDim, steps int) (float32, time.Duration) {

	heads := 4
	if dim >= 256 {
		heads = 8
	}
	if dim >= 512 {
		heads = 16
	}
	kvHeads := max(1, heads/2)
	headDim := dim / heads
	kvDim := kvHeads * headDim
	vocabSize := 256
	seqLen := 64

	if graph == nil {
		return runSweepTrialExec(dataPath, lr, dim, layers, ffnDim, steps)
	}

	ropeTheta := 10000.0
	ret := graph.BuildFullGraph(dim, kvDim, headDim, heads, kvHeads, ffnDim,
		vocabSize, layers, seqLen, ropeTheta, 1)
	if ret != 0 {
		return 999.0, 0
	}

	// Initialize weights
	nW := graph.GraphNumWeights()
	kaiming := func(rows, cols int) []float32 {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return d
	}
	ones := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = 1.0
		}
		return s
	}

	embed := make([]float32, vocabSize*dim)
	for i := range embed {
		embed[i] = float32(rand.NormFloat64()) * 0.02
	}

	var paramData [][]float32
	paramData = append(paramData, embed, ones(dim))
	for range layers {
		paramData = append(paramData,
			ones(dim), kaiming(dim, dim), kaiming(kvDim, dim), kaiming(kvDim, dim),
			kaiming(dim, dim), make([]float32, dim), make([]float32, kvDim), make([]float32, kvDim),
			ones(dim), kaiming(ffnDim, dim), kaiming(ffnDim, dim), kaiming(dim, ffnDim),
		)
	}

	for i := 0; i < nW && i < len(paramData); i++ {
		graph.GraphSetVariable(i, paramData[i])
	}
	graph.Sync()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	n := seqLen + 1
	curLR := float32(lr)

	t0 := time.Now()
	var lastLoss float32
	for step := 1; step <= steps; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokens := make([]int32, n)
		targets := make([]int32, n)
		for i := 0; i < n; i++ {
			tokens[i] = int32(data[start+i])
			targets[i] = int32(data[start+i+1])
		}
		lastLoss = graph.GraphTrainStepAdam(tokens, targets, curLR)
	}
	graph.Sync()

	return lastLoss, time.Since(t0)
}

// CPU fallback for sweep when no GraphTrainEngine is available
func runSweepTrialCPU(eng mongoose.Engine, data []int,
	lr float64, dim, heads, kvHeads, layers, ffnDim, vocabSize, seqLen, steps int) (float32, time.Duration) {
	// Simplified CPU training loop — just measure convergence trend
	rng := rand.New(rand.NewSource(42))
	n := seqLen

	// Random "model" — just track loss via simple bigram statistics
	bigram := make([]float32, vocabSize*vocabSize)
	for i := range bigram {
		bigram[i] = 1.0 / float32(vocabSize)
	}

	t0 := time.Now()
	var lastLoss float32
	curLR := float32(lr)

	for step := 0; step < steps; step++ {
		start := rng.Intn(len(data) - n - 1)
		var stepLoss float64
		for i := 0; i < n-1; i++ {
			src := data[start+i]
			tgt := data[start+i+1]
			p := bigram[src*vocabSize+tgt]
			if p < 1e-10 {
				p = 1e-10
			}
			stepLoss += -math.Log(float64(p))
			// SGD update on bigram
			bigram[src*vocabSize+tgt] += curLR * (1.0 - p)
			// Renormalize row
			var sum float32
			for j := 0; j < vocabSize; j++ {
				if bigram[src*vocabSize+j] < 0 {
					bigram[src*vocabSize+j] = 0
				}
				sum += bigram[src*vocabSize+j]
			}
			if sum > 0 {
				for j := 0; j < vocabSize; j++ {
					bigram[src*vocabSize+j] /= sum
				}
			}
		}
		lastLoss = float32(stepLoss / float64(n-1))
	}

	return lastLoss, time.Since(t0)
}

var floorRe = regexp.MustCompile(`floor[=:]?\s*([\d.]+)`)

// runSweepTrialExec runs a trial by execing `ai train` as a subprocess.
// Works on any backend — the train command handles routing.
func runSweepTrialExec(dataPath string, lr float64, dim, layers, ffnDim, steps int) (float32, time.Duration) {
	exe, err := os.Executable()
	if err != nil {
		log.Printf("[sweep] cannot find self: %v", err)
		return 999.0, 0
	}

	args := []string{"train",
		fmt.Sprintf("data=%s", dataPath),
		fmt.Sprintf("--dim=%d", dim),
		fmt.Sprintf("--layers=%d", layers),
		fmt.Sprintf("--ffn-dim=%d", ffnDim),
		fmt.Sprintf("--steps=%d", steps),
		fmt.Sprintf("--lr=%e", lr),
	}

	t0 := time.Now()
	cmd := exec.Command(exe, args...)
	out, err := cmd.CombinedOutput()
	elapsed := time.Since(t0)

	if err != nil {
		log.Printf("[sweep] trial failed: %v\n%s", err, string(out))
		return 999.0, elapsed
	}

	// Parse "floor=X.XXX" or "floor: X.XXX" from output
	matches := floorRe.FindSubmatch(out)
	if len(matches) < 2 {
		log.Printf("[sweep] could not parse loss from output")
		return 999.0, elapsed
	}
	loss, err := strconv.ParseFloat(string(matches[1]), 32)
	if err != nil {
		return 999.0, elapsed
	}
	return float32(loss), elapsed
}

func parseFloatList(s string, def []float64) []float64 {
	if s == "" {
		return def
	}
	parts := strings.Split(s, ",")
	var result []float64
	for _, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err == nil {
			result = append(result, v)
		}
	}
	if len(result) == 0 {
		return def
	}
	return result
}

func parseIntList(s string, def []int) []int {
	if s == "" {
		return def
	}
	parts := strings.Split(s, ",")
	var result []int
	for _, p := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err == nil {
			result = append(result, v)
		}
	}
	if len(result) == 0 {
		return def
	}
	return result
}
